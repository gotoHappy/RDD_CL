"""
UMAP / t-SNE visualization of patch features to test the
"generic anomaly prior" hypothesis.

Hypothesis
----------
After contrastive training, *anomalous* patches from different object
categories should cluster together in feature space — even though training
saw only mydata's industrial parts and three defect types. This would
imply the projector learned a generic "anomaly direction" rather than per-
object features. If the hypothesis holds, we expect:

  - In the trained (contrastive) embedding: anomalous patches form a
    coherent cluster, distinct from normal patches.
  - The clustering crosses category boundaries — an anomalous patch from
    Filter is closer to an anomalous patch from Teapot than to normal
    Filter patches.
  - The raw DINOv3 (baseline) embedding does NOT show this structure
    (anomalous patches are scattered among their per-category clusters).

Method
------
For every testdata sample we:
  1. Encode (ref, query) with both the contrastive and baseline models.
  2. Pool gt to the patch grid resolution, label each query patch
     anomalous / normal / ignore via thresholds.
  3. Per category, sample a balanced number of anomalous and normal
     patches to avoid the long-tail imbalance dominating the plot.

Outputs (saved under ``--output``):
  - ``embedding_comparison.png``   — 2×2 UMAP figure (baseline vs contrastive,
                                     coloured by anomaly status / category).
  - ``metrics.json``               — linear-probe AUROC, kNN purity,
                                     cross-category anomaly purity.
  - ``features.npz``               — raw collected features for further analysis.

Usage
-----
    python scripts/visualize_embedding.py <checkpoint.pth> \\
        --testdata-root testdata \\
        --output outputs/.../embedding \\
        [--per-cat-samples 200]    # patches per (category, label)
        [--method umap|tsne]
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvff

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.models as models
from robust_defect_detection import utils

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_image(path, figsize):
    img = Image.open(path).convert("RGB")
    img = img.resize((figsize[1], figsize[0]), Image.BILINEAR)
    t = tvff.to_tensor(img)
    return tvff.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD).unsqueeze(0)


def _list_testdata_samples(root):
    root = Path(root)
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sample_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            ref_p = sample_dir / "ref.png"
            qry_p = sample_dir / "query.png"
            gt_p = sample_dir / "gt.png"
            if ref_p.exists() and qry_p.exists() and gt_p.exists():
                yield cat_dir.name, sample_dir.name, sample_dir


# ---------------------------------------------------------------------------
# Patch feature collection
# ---------------------------------------------------------------------------

def collect_patch_features(
    testdata_root, contrastive_model, baseline_model,
    figsize, device, layer_idx=-1,
    defect_thresh=0.3, normal_thresh=0.05,
    per_cat_anom=200, per_cat_norm=200,
    rng_seed=0,
):
    """For every sample, run inference and collect per-patch feature vectors
    + anomaly label + category. Subsample per category to a balanced count.

    Returns:
        feats_c (N, D_c) contrastive features  — L2-normalised
        feats_b (N, D_b) baseline (raw DINOv3) features — L2-normalised
        labels  (N,)     0 = normal, 1 = anomalous
        cats    (N,)     str category names
    """
    rng = np.random.default_rng(rng_seed)

    # First pass: collect ALL anomalous + normal patches per category
    raw = defaultdict(lambda: {"anom_c": [], "anom_b": [], "norm_c": [], "norm_b": []})

    samples_by_cat = defaultdict(list)
    for cat, name, sd in _list_testdata_samples(testdata_root):
        samples_by_cat[cat].append((name, sd))

    total = sum(len(v) for v in samples_by_cat.values())
    print(f"[viz] processing {total} samples across {len(samples_by_cat)} categories")

    for cat, samples in samples_by_cat.items():
        print(f"  [{cat}] {len(samples)} samples …")
        for name, sdir in samples:
            ref_t = _load_image(sdir / "ref.png", figsize).to(device)
            qry_t = _load_image(sdir / "query.png", figsize).to(device)

            gt_arr = np.array(Image.open(sdir / "gt.png").convert("L"))
            gt_bin = (gt_arr > 0).astype(np.float32)

            with torch.no_grad():
                # Contrastive (post-projector)
                _, q_feats_c = contrastive_model.encode_pair(ref_t, qry_t)
                # Baseline (raw DINOv3 features at the same layer; LoRA B=0 ⇒ no effect)
                _, q_feats_b = baseline_model.backbone(ref_t, qry_t)

            feat_c = q_feats_c[layer_idx].squeeze(0)  # (C_c, h, w)
            feat_b = q_feats_b[layer_idx].squeeze(0)  # (C_b, h, w)
            h, w = feat_c.shape[-2:]

            # Pool gt to the feature grid
            gt_t = torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0)
            gt_pooled = F.adaptive_avg_pool2d(gt_t, (h, w)).squeeze().numpy()
            anom_mask = (gt_pooled > defect_thresh).ravel()
            norm_mask = (gt_pooled < normal_thresh).ravel()

            feat_c_flat = feat_c.permute(1, 2, 0).reshape(h * w, -1).cpu().numpy()
            feat_b_flat = feat_b.permute(1, 2, 0).reshape(h * w, -1).cpu().numpy()

            if anom_mask.any():
                raw[cat]["anom_c"].append(feat_c_flat[anom_mask])
                raw[cat]["anom_b"].append(feat_b_flat[anom_mask])
            if norm_mask.any():
                raw[cat]["norm_c"].append(feat_c_flat[norm_mask])
                raw[cat]["norm_b"].append(feat_b_flat[norm_mask])

    # Second pass: per-category balanced subsampling
    out_c, out_b, out_labels, out_cats = [], [], [], []

    for cat in sorted(raw.keys()):
        for kind, label, target in [("anom", 1, per_cat_anom), ("norm", 0, per_cat_norm)]:
            chunks_c = raw[cat][f"{kind}_c"]
            chunks_b = raw[cat][f"{kind}_b"]
            if not chunks_c:
                print(f"    [{cat}] {kind}: 0 patches — skipped")
                continue
            arr_c = np.concatenate(chunks_c)
            arr_b = np.concatenate(chunks_b)
            n_avail = len(arr_c)
            n_take = min(target, n_avail)
            idx = rng.choice(n_avail, n_take, replace=False)
            out_c.append(arr_c[idx])
            out_b.append(arr_b[idx])
            out_labels.extend([label] * n_take)
            out_cats.extend([cat] * n_take)
            print(f"    [{cat}] {kind}: {n_take}/{n_avail} patches")

    feats_c = np.concatenate(out_c).astype(np.float32)
    feats_b = np.concatenate(out_b).astype(np.float32)
    labels = np.array(out_labels, dtype=np.int64)
    cats = np.array(out_cats)
    print(f"[viz] collected total {len(labels)} patches "
          f"(anom={int((labels == 1).sum())}, norm={int((labels == 0).sum())})")
    return feats_c, feats_b, labels, cats


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_to_2d(X, method="umap", random_state=0):
    """2D reduction. Falls back to t-SNE if umap-learn is unavailable."""
    method = method.lower()
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2, n_neighbors=15, min_dist=0.1,
                metric="cosine", random_state=random_state,
            )
            return reducer.fit_transform(X), "UMAP"
        except ImportError:
            warnings.warn("umap-learn not installed, falling back to t-SNE")
            method = "tsne"
    from sklearn.manifold import TSNE
    perplexity = max(5, min(30, len(X) // 4))
    reducer = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        metric="cosine", random_state=random_state, n_iter=1000,
    )
    return reducer.fit_transform(X), "t-SNE"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_embeddings(emb_b, emb_c, labels, cats, out_path, method_name="UMAP"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ---- Top row: colour by anomaly status ----
    n_norm, n_anom = int((labels == 0).sum()), int((labels == 1).sum())
    for ax, emb, name in [
        (axes[0, 0], emb_b, "Baseline (raw DINOv3)"),
        (axes[0, 1], emb_c, "Contrastive (trained)"),
    ]:
        norm_m = labels == 0
        anom_m = labels == 1
        ax.scatter(emb[norm_m, 0], emb[norm_m, 1], c="steelblue",
                   s=10, alpha=0.4, edgecolors="none", label=f"normal (n={n_norm})")
        ax.scatter(emb[anom_m, 0], emb[anom_m, 1], c="crimson",
                   s=14, alpha=0.75, edgecolors="none", label=f"anomalous (n={n_anom})")
        ax.set_title(f"{name} — coloured by anomaly")
        ax.set_xlabel(f"{method_name} dim 1"); ax.set_ylabel(f"{method_name} dim 2")
        ax.legend(loc="best", framealpha=0.85)

    # ---- Bottom row: colour by category, marker by anomaly ----
    unique_cats = sorted(set(cats))
    cmap = plt.get_cmap("tab10")
    cat_color = {c: cmap(i % 10) for i, c in enumerate(unique_cats)}

    for ax, emb, name in [
        (axes[1, 0], emb_b, "Baseline (raw DINOv3)"),
        (axes[1, 1], emb_c, "Contrastive (trained)"),
    ]:
        for c in unique_cats:
            m_norm = (cats == c) & (labels == 0)
            m_anom = (cats == c) & (labels == 1)
            color = [cat_color[c]]
            if m_norm.any():
                ax.scatter(emb[m_norm, 0], emb[m_norm, 1], c=color,
                           s=8, alpha=0.25, edgecolors="none", marker="o")
            if m_anom.any():
                ax.scatter(emb[m_anom, 0], emb[m_anom, 1], c=color,
                           s=28, alpha=0.85, edgecolors="black",
                           linewidth=0.4, marker="^", label=f"{c}")
        ax.set_title(f"{name} — coloured by category, △=anomalous, ◯=normal")
        ax.set_xlabel(f"{method_name} dim 1"); ax.set_ylabel(f"{method_name} dim 2")
        ax.legend(loc="best", fontsize=8, framealpha=0.85, title="category (△ markers)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[viz] embedding figure saved → {out_path}")


# ---------------------------------------------------------------------------
# Quantitative metrics
# ---------------------------------------------------------------------------

def linear_probe_auroc(feats, labels, n_splits=5, seed=0):
    """5-fold cross-validated AUROC of a linear classifier separating
    normal vs anomalous patches. Higher = more linearly separable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(feats, labels):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(feats[tr], labels[tr])
        probs = clf.predict_proba(feats[te])[:, 1]
        aucs.append(roc_auc_score(labels[te], probs))
    return float(np.mean(aucs)), float(np.std(aucs))


def knn_purity(feats, labels, k=10):
    """For each patch, fraction of k nearest neighbours that share its
    label (anomalous / normal). Reports overall + per-class purity."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(feats)
    _, neighbors = nn.kneighbors(feats)
    same = (labels[neighbors[:, 1:]] == labels[:, None]).mean(axis=1)
    return {
        "overall": float(same.mean()),
        "anom_only": float(same[labels == 1].mean()) if (labels == 1).any() else None,
        "norm_only": float(same[labels == 0].mean()) if (labels == 0).any() else None,
    }


def cross_category_anomaly_purity(feats, labels, cats, k=10):
    """For each anomalous patch, fraction of its k nearest neighbours that
    are anomalous AND from a *different* category. High value → trained
    embedding has merged anomalies across categories (generic prior).
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(feats)
    _, neighbors = nn.kneighbors(feats)

    anom_idx = np.where(labels == 1)[0]
    if len(anom_idx) == 0:
        return None
    rates = []
    for i in anom_idx:
        nbrs = neighbors[i, 1:]
        is_cross_cat_anom = (labels[nbrs] == 1) & (cats[nbrs] != cats[i])
        rates.append(is_cross_cat_anom.mean())
    return float(np.mean(rates))


def compute_all_metrics(feats, labels, cats, k=10):
    auroc_mean, auroc_std = linear_probe_auroc(feats, labels)
    knn = knn_purity(feats, labels, k=k)
    cross = cross_category_anomaly_purity(feats, labels, cats, k=k)
    return {
        "linear_probe_auroc_mean": auroc_mean,
        "linear_probe_auroc_std": auroc_std,
        "knn_purity_overall": knn["overall"],
        "knn_purity_anom_only": knn["anom_only"],
        "knn_purity_norm_only": knn["norm_only"],
        "cross_category_anom_neighbour_rate": cross,
    }


def print_metrics_table(metrics_b, metrics_c):
    rows = [
        ("Linear-probe AUROC (5-fold)",       "linear_probe_auroc_mean", True),
        ("kNN purity — overall",               "knn_purity_overall",       False),
        ("kNN purity — anomalous patches",     "knn_purity_anom_only",     False),
        ("kNN purity — normal patches",        "knn_purity_norm_only",     False),
        ("Cross-category anomaly nbr rate",    "cross_category_anom_neighbour_rate", False),
    ]
    print("\n" + "=" * 80)
    print("  Embedding metrics — baseline (raw DINOv3) vs contrastive (trained)")
    print("=" * 80)
    print(f"  {'Metric':<40} {'Baseline':>12} {'Contrastive':>14} {'Δ (C − B)':>12}")
    print("-" * 80)
    for label, key, with_std in rows:
        b = metrics_b.get(key)
        c = metrics_c.get(key)
        if b is None or c is None:
            print(f"  {label:<40} {'N/A':>12} {'N/A':>14} {'N/A':>12}")
            continue
        if with_std:
            std_b = metrics_b.get("linear_probe_auroc_std", 0.0)
            std_c = metrics_c.get("linear_probe_auroc_std", 0.0)
            print(f"  {label:<40} {b:>8.4f}±{std_b:.3f} {c:>10.4f}±{std_c:.3f} {c - b:>+12.4f}")
        else:
            print(f"  {label:<40} {b:>12.4f} {c:>14.4f} {c - b:>+12.4f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--testdata-root", type=str, default="testdata")
    p.add_argument("--output", type=str, default="outputs/embedding")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    p.add_argument("--per-cat-anom", type=int, default=200,
                   help="Anomalous patches sampled per category.")
    p.add_argument("--per-cat-norm", type=int, default=200,
                   help="Normal patches sampled per category.")
    p.add_argument("--layer-idx", type=int, default=-1,
                   help="Which transformer layer to use (default: last).")
    p.add_argument("--defect-thresh", type=float, default=0.3,
                   help="patch-level pooled gt > this → anomalous.")
    p.add_argument("--normal-thresh", type=float, default=0.05,
                   help="patch-level pooled gt < this → normal.")
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(utils.resolve_path(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # ---- Models ----
    print("[viz] loading checkpoint …")
    contrastive_model, ckpt_data = models.load_checkpoint_model(
        args.checkpoint, device=device, verbose=False)
    contrastive_model.eval()

    print("[viz] building baseline (LoRA B=0 ⇔ raw DINOv3) …")
    baseline_model = models.get_model(**ckpt_data["args"]["model"])
    baseline_model = models.wrap_model_for_gpus(baseline_model, device=device)
    baseline_model.eval()

    figsize = ckpt_data["args"]["dataset"]["figsize"]

    # ---- Collect patches ----
    feats_c, feats_b, labels, cats = collect_patch_features(
        utils.resolve_path(args.testdata_root),
        contrastive_model, baseline_model,
        figsize=figsize, device=device,
        layer_idx=args.layer_idx,
        defect_thresh=args.defect_thresh,
        normal_thresh=args.normal_thresh,
        per_cat_anom=args.per_cat_anom,
        per_cat_norm=args.per_cat_norm,
        rng_seed=args.seed,
    )

    np.savez(
        out_dir / "features.npz",
        feats_c=feats_c, feats_b=feats_b, labels=labels, cats=cats,
    )

    # ---- 2D reduction ----
    print(f"[viz] reducing to 2D with {args.method} …")
    emb_b, method_name = reduce_to_2d(feats_b, method=args.method, random_state=args.seed)
    emb_c, _ = reduce_to_2d(feats_c, method=args.method, random_state=args.seed)

    plot_embeddings(emb_b, emb_c, labels, cats,
                    str(out_dir / "embedding_comparison.png"),
                    method_name=method_name)

    # ---- Quantitative metrics ----
    print("[viz] computing metrics …")
    metrics_b = compute_all_metrics(feats_b, labels, cats, k=args.knn_k)
    metrics_c = compute_all_metrics(feats_c, labels, cats, k=args.knn_k)
    metrics = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "testdata_root": str(args.testdata_root),
            "layer_idx": args.layer_idx,
            "per_cat_anom": args.per_cat_anom,
            "per_cat_norm": args.per_cat_norm,
            "knn_k": args.knn_k,
            "n_anom": int((labels == 1).sum()),
            "n_norm": int((labels == 0).sum()),
            "feat_dim_baseline": int(feats_b.shape[1]),
            "feat_dim_contrastive": int(feats_c.shape[1]),
        },
        "baseline": metrics_b,
        "contrastive": metrics_c,
    }
    with open(out_dir / "metrics.json", "w") as fd:
        json.dump(metrics, fd, indent=2)

    print_metrics_table(metrics_b, metrics_c)
    print(f"\n[viz] all outputs in {out_dir}")


if __name__ == "__main__":
    main()
