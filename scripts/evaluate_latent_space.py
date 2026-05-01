"""Layer-wise latent-space metrics for raw DINOv3 vs contrastive embeddings.

This is the quantitative-only counterpart of ``visualize_embedding.py``.
It does not run UMAP/t-SNE and does not save figures. For every layer listed
in ``checkpoint['args']['model']['layers']`` it reports:

  - Linear-probe AUROC (5-fold CV): logreg on patch embeddings for anom/norm
  - kNN purity (overall): fraction of k neighbours with the same patch label
  - kNN purity (anom only): same-label neighbour fraction for anomaly patches
  - Cross-category anom neighbour rate: for anomaly patches, fraction of k
    neighbours that are also anomalous and from a different category

Usage
-----
::

    python scripts/evaluate_latent_space.py outputs/.../best.pth \
        --testdata-root testdata \
        --output outputs/.../latent_space_metrics
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvff
from tqdm.auto import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.models as models
from robust_defect_detection import utils


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--testdata-root", type=str, default="testdata")
    p.add_argument("--output", type=str, default="outputs/latent_space_metrics")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--per-cat-anom", type=int, default=200)
    p.add_argument("--per-cat-norm", type=int, default=200)
    p.add_argument("--defect-thresh", type=float, default=0.3)
    p.add_argument("--normal-thresh", type=float, default=0.05)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-features", action="store_true")
    return p.parse_args()


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


def _unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _append_layer_features(store, cat, layer_id, kind, feat_c, feat_b, mask):
    if not mask.any():
        return
    store[cat][int(layer_id)][f"{kind}_c"].append(feat_c[mask])
    store[cat][int(layer_id)][f"{kind}_b"].append(feat_b[mask])


@torch.no_grad()
def collect_layer_patch_features(
    testdata_root,
    contrastive_model,
    baseline_model,
    layers,
    figsize,
    device,
    defect_thresh=0.3,
    normal_thresh=0.05,
    per_cat_anom=200,
    per_cat_norm=200,
    rng_seed=0,
):
    """Collect balanced patch features per category/label for every layer.

    Returns a dict:
        layer_id -> {
            "contrastive": (N, D_c),
            "baseline": (N, D_b),
            "labels": (N,),
            "cats": (N,)
        }
    """
    rng = np.random.default_rng(rng_seed)
    store = defaultdict(
        lambda: defaultdict(
            lambda: {"anom_c": [], "anom_b": [], "norm_c": [], "norm_b": []}
        )
    )

    samples_by_cat = defaultdict(list)
    for cat, name, sd in _list_testdata_samples(testdata_root):
        samples_by_cat[cat].append((name, sd))
    total = sum(len(v) for v in samples_by_cat.values())
    if total == 0:
        raise SystemExit(f"No ref/query/gt test samples found under {testdata_root}")

    print(
        f"[latent] processing {total} samples across "
        f"{len(samples_by_cat)} categories"
    )

    c_model = _unwrap(contrastive_model)
    b_model = _unwrap(baseline_model)

    for cat, samples in samples_by_cat.items():
        print(f"  [{cat}] {len(samples)} samples")
        for _, sdir in tqdm(samples, desc=cat, leave=False, dynamic_ncols=True):
            ref_t = _load_image(sdir / "ref.png", figsize).to(device)
            qry_t = _load_image(sdir / "query.png", figsize).to(device)

            gt_arr = np.array(Image.open(sdir / "gt.png").convert("L"))
            gt_bin = (gt_arr > 0).astype(np.float32)

            _, q_feats_c = c_model.encode_pair(ref_t, qry_t)
            _, q_feats_b = b_model.backbone(ref_t, qry_t)

            for layer_id, feat_c_t, feat_b_t in zip(layers, q_feats_c, q_feats_b):
                feat_c = feat_c_t.squeeze(0)
                feat_b = feat_b_t.squeeze(0)
                h, w = feat_c.shape[-2:]

                gt_t = torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0)
                gt_pooled = F.adaptive_avg_pool2d(gt_t, (h, w)).squeeze().numpy()
                anom_mask = (gt_pooled > defect_thresh).ravel()
                norm_mask = (gt_pooled < normal_thresh).ravel()

                feat_c_flat = (
                    feat_c.permute(1, 2, 0).reshape(h * w, -1).detach().cpu().numpy()
                )
                feat_b_flat = (
                    feat_b.permute(1, 2, 0).reshape(h * w, -1).detach().cpu().numpy()
                )
                _append_layer_features(
                    store, cat, layer_id, "anom", feat_c_flat, feat_b_flat, anom_mask
                )
                _append_layer_features(
                    store, cat, layer_id, "norm", feat_c_flat, feat_b_flat, norm_mask
                )

    layer_data = {}
    for layer_id in layers:
        out_c, out_b, out_labels, out_cats = [], [], [], []
        print(f"[latent] layer {layer_id}: balanced sampling")
        for cat in sorted(store.keys()):
            layer_store = store[cat][int(layer_id)]
            for kind, label, target in (
                ("anom", 1, per_cat_anom),
                ("norm", 0, per_cat_norm),
            ):
                chunks_c = layer_store[f"{kind}_c"]
                chunks_b = layer_store[f"{kind}_b"]
                if not chunks_c:
                    print(f"    [{cat}] {kind}: 0 patches")
                    continue
                arr_c = np.concatenate(chunks_c).astype(np.float32)
                arr_b = np.concatenate(chunks_b).astype(np.float32)
                n_avail = len(arr_c)
                n_take = min(int(target), n_avail)
                idx = rng.choice(n_avail, n_take, replace=False)
                out_c.append(arr_c[idx])
                out_b.append(arr_b[idx])
                out_labels.extend([label] * n_take)
                out_cats.extend([cat] * n_take)
                print(f"    [{cat}] {kind}: {n_take}/{n_avail} patches")

        if not out_c:
            raise SystemExit(f"No patch features collected for layer {layer_id}")
        labels = np.array(out_labels, dtype=np.int64)
        cats = np.array(out_cats)
        layer_data[int(layer_id)] = {
            "contrastive": np.concatenate(out_c).astype(np.float32),
            "baseline": np.concatenate(out_b).astype(np.float32),
            "labels": labels,
            "cats": cats,
        }
        print(
            f"[latent] layer {layer_id}: total={len(labels)} "
            f"anom={int((labels == 1).sum())} norm={int((labels == 0).sum())}"
        )

    return layer_data


def linear_probe_auroc(feats, labels, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2 or counts.min() < 2:
        return None, None, 0
    folds = min(int(n_splits), int(counts.min()))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    aucs = []
    for tr, te in skf.split(feats, labels):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        )
        clf.fit(feats[tr], labels[tr])
        probs = clf.predict_proba(feats[te])[:, 1]
        aucs.append(roc_auc_score(labels[te], probs))
    return float(np.mean(aucs)), float(np.std(aucs)), folds


def knn_metrics(feats, labels, cats, k=10):
    from sklearn.neighbors import NearestNeighbors

    if len(feats) < 2:
        return {
            "knn_k_effective": 0,
            "knn_purity_overall": None,
            "knn_purity_anom_only": None,
            "cross_category_anom_neighbour_rate": None,
        }

    k_eff = min(int(k), len(feats) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(feats)
    _, neighbors = nn.kneighbors(feats)
    neighbors = neighbors[:, 1:]

    same_label = labels[neighbors] == labels[:, None]
    purity_per_patch = same_label.mean(axis=1)
    anom_mask = labels == 1

    cross = None
    if anom_mask.any():
        rates = []
        for i in np.where(anom_mask)[0]:
            nbrs = neighbors[i]
            cross_anom = (labels[nbrs] == 1) & (cats[nbrs] != cats[i])
            rates.append(cross_anom.mean())
        cross = float(np.mean(rates))

    return {
        "knn_k_effective": int(k_eff),
        "knn_purity_overall": float(purity_per_patch.mean()),
        "knn_purity_anom_only": (
            float(purity_per_patch[anom_mask].mean()) if anom_mask.any() else None
        ),
        "cross_category_anom_neighbour_rate": cross,
    }


def compute_metrics(feats, labels, cats, k=10, seed=0):
    auc_mean, auc_std, folds = linear_probe_auroc(feats, labels, n_splits=5, seed=seed)
    out = {
        "linear_probe_auroc_mean": auc_mean,
        "linear_probe_auroc_std": auc_std,
        "linear_probe_cv_folds": folds,
    }
    out.update(knn_metrics(feats, labels, cats, k=k))
    return out


def _fmt(v):
    return "nan" if v is None else f"{float(v):.4f}"


def print_layer_table(rows):
    print("\n" + "=" * 112)
    print("Layer-wise latent-space metrics")
    print("=" * 112)
    print(
        f"{'method':<12} {'layer':>5} {'AUROC':>8} {'kNN-all':>9} "
        f"{'kNN-anom':>9} {'cross-cat-anom':>15} {'N':>7}"
    )
    print("-" * 112)
    for row in rows:
        print(
            f"{row['method']:<12} {row['layer']:>5} "
            f"{_fmt(row['linear_probe_auroc_mean']):>8} "
            f"{_fmt(row['knn_purity_overall']):>9} "
            f"{_fmt(row['knn_purity_anom_only']):>9} "
            f"{_fmt(row['cross_category_anom_neighbour_rate']):>15} "
            f"{row['num_patches']:>7}"
        )
    print("=" * 112)


def write_outputs(out_dir, payload, rows, layer_data=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latent_space_metrics_by_layer.json"
    csv_path = out_dir / "latent_space_metrics_by_layer.csv"
    with open(json_path, "w") as fd:
        json.dump(payload, fd, indent=2)

    fieldnames = [
        "method",
        "layer",
        "linear_probe_auroc_mean",
        "linear_probe_auroc_std",
        "linear_probe_cv_folds",
        "knn_k_effective",
        "knn_purity_overall",
        "knn_purity_anom_only",
        "cross_category_anom_neighbour_rate",
        "num_patches",
        "num_anom_patches",
        "num_norm_patches",
        "feat_dim",
    ]
    with open(csv_path, "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    npz_path = None
    if layer_data is not None:
        arrays = {}
        for layer_id, data in layer_data.items():
            prefix = f"layer_{layer_id}"
            arrays[f"{prefix}_contrastive"] = data["contrastive"]
            arrays[f"{prefix}_baseline"] = data["baseline"]
            arrays[f"{prefix}_labels"] = data["labels"]
            arrays[f"{prefix}_cats"] = data["cats"]
        npz_path = out_dir / "latent_space_features_by_layer.npz"
        np.savez(npz_path, **arrays)
    return json_path, csv_path, npz_path


def main():
    args = parse_args()
    out_dir = Path(utils.resolve_path(args.output))
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    print("[latent] loading checkpoint")
    contrastive_model, ckpt = models.load_checkpoint_model(
        args.checkpoint, device=device, verbose=False
    )
    contrastive_model.eval()

    model_args = ckpt["args"]["model"]
    if model_args.get("name") != "dino2 + contrast_learning":
        raise SystemExit(
            "evaluate_latent_space.py only supports 'dino2 + contrast_learning' checkpoints."
        )
    layers = [int(layer) for layer in model_args["layers"]]
    figsize = tuple(int(v) for v in ckpt["args"]["dataset"]["figsize"])

    print("[latent] building raw DINOv3 baseline")
    baseline_model = models.get_model(**model_args)
    baseline_model = models.wrap_model_for_gpus(baseline_model, device=device)
    baseline_model.eval()

    layer_data = collect_layer_patch_features(
        testdata_root=utils.resolve_path(args.testdata_root),
        contrastive_model=contrastive_model,
        baseline_model=baseline_model,
        layers=layers,
        figsize=figsize,
        device=device,
        defect_thresh=args.defect_thresh,
        normal_thresh=args.normal_thresh,
        per_cat_anom=args.per_cat_anom,
        per_cat_norm=args.per_cat_norm,
        rng_seed=args.seed,
    )

    rows = []
    metrics_by_layer = {}
    for layer_id in layers:
        data = layer_data[int(layer_id)]
        labels = data["labels"]
        cats = data["cats"]
        metrics_by_layer[str(layer_id)] = {}
        for method, feat_key in (("raw_dino", "baseline"), ("trained", "contrastive")):
            feats = data[feat_key]
            print(f"[latent] metrics: {method}, layer {layer_id}")
            metrics = compute_metrics(
                feats,
                labels,
                cats,
                k=args.knn_k,
                seed=args.seed,
            )
            metrics_by_layer[str(layer_id)][method] = metrics
            rows.append({
                "method": method,
                "layer": int(layer_id),
                **metrics,
                "num_patches": int(len(labels)),
                "num_anom_patches": int((labels == 1).sum()),
                "num_norm_patches": int((labels == 0).sum()),
                "feat_dim": int(feats.shape[1]),
            })

    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "testdata_root": str(args.testdata_root),
            "layers": layers,
            "per_cat_anom": int(args.per_cat_anom),
            "per_cat_norm": int(args.per_cat_norm),
            "defect_thresh": float(args.defect_thresh),
            "normal_thresh": float(args.normal_thresh),
            "knn_k": int(args.knn_k),
            "seed": int(args.seed),
        },
        "metrics_by_layer": metrics_by_layer,
        "rows": rows,
    }

    json_path, csv_path, npz_path = write_outputs(
        out_dir,
        payload,
        rows,
        layer_data=layer_data if args.save_features else None,
    )

    print_layer_table(rows)
    print(f"\n[latent] wrote {json_path}")
    print(f"[latent] wrote {csv_path}")
    if npz_path is not None:
        print(f"[latent] wrote {npz_path}")


if __name__ == "__main__":
    main()
