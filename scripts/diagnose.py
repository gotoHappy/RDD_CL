"""
Diagnostic script for comparing contrastive (trained) vs baseline (untrained
DINO) inference on testdata.

Inputs:
    testdata/{Category}/{Category}_{N}/{ref.png, query.png, gt.png, heatmap.png}

Outputs (under ``--output``):
    1. ``training_curves.png``         — training pos_sim / hard_neg_sim / loss
    2. ``testdata_per_category.csv``   — image- and pixel-level AUROC per category
                                         + MACRO_MEAN / MICRO_OVERALL summary rows
    3. ``testdata_per_sample.csv``     — per-sample pixel AUROC + max-score
    4. ``visualizations/{Cat}/{Cat}_{N}.png`` — 2×3 figure for every sample:
           Row 1: ref, query, gt
           Row 2: PIAD heatmap.png, DINO baseline, Contrastive (trained)
           score maps use independent per-image min-max normalisation

Usage:
    python scripts/diagnose.py <checkpoint.pth> \\
        --testdata-root testdata \\
        --output outputs/diagnose

note: ``compute_score_map`` applies a gamma=2 suppression to small cosine
distances; the ``defect_thresh`` boundary it uses is read from the saved
training config (margin-loss / contrastive-loss → patch-defect-thresh).
"""

import argparse
import os
import pickle
import sys
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


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_score_map(
    encode_fn,
    ref_img,
    query_img,
    target_shp,
    trained_layers,
    smooth_cfg,
    defect_thresh=0.3,
    gamma=2,
    use_layers=None,
    aggregation="mean",
    top_k=3,
    zscore_stats=None,
    stats_key=None,
):
    """Cosine-distance score map for the contrastive model (encode_fn returns
    a per-layer feature pair). Applied to both the trained projector branch
    and the raw DINO baseline branch.

    The ``defect_thresh`` + ``gamma`` post-processing suppresses small cosine
    distances (likely noise) while preserving large ones (high-confidence
    anomaly), as discussed in 实验结果分析.md.
    """
    layer_maps = compute_layer_score_maps(
        encode_fn,
        ref_img,
        query_img,
        target_shp,
        defect_thresh=defect_thresh,
        gamma=gamma,
    )

    selected_layers = trained_layers
    if trained_layers is not None:
        trained_layers = [int(layer) for layer in trained_layers]
        selected_layers = trained_layers if use_layers is None else [int(layer) for layer in use_layers]
        layer_map_by_id = {
            layer: layer_map for layer, layer_map in zip(trained_layers, layer_maps)
        }
        layer_maps = [layer_map_by_id[layer] for layer in selected_layers]

    if zscore_stats is not None:
        layer_maps = _apply_zscore_to_layer_maps(
            layer_maps=layer_maps,
            selected_layers=selected_layers,
            zscore_stats=zscore_stats,
            stats_key=stats_key,
        )

    fused = fuse_layer_maps(layer_maps, aggregation=aggregation, top_k=top_k)
    k = smooth_cfg.get("gaussian-kernel", 5)
    if k and k > 1:
        sig = smooth_cfg.get("gaussian-sigma", 1.0)
        fused = tvff.gaussian_blur(fused.unsqueeze(1), [k, k], [sig, sig]).squeeze(1)
    return fused.squeeze(0).detach().cpu().numpy()  # (H, W)


def compute_layer_score_maps(
    encode_fn,
    ref_img,
    query_img,
    target_shp,
    defect_thresh=0.3,
    gamma=2,
):
    """Return one post-processed cosine-distance score map per feature layer."""
    ref_features, query_features = encode_fn(ref_img, query_img)
    layer_maps = []
    d = 1 - defect_thresh
    for rf, qf in zip(ref_features, query_features):
        score = 1.0 - F.cosine_similarity(rf, qf, dim=1)
        score = d * (score / d).pow(gamma)
        score = F.interpolate(
            score.unsqueeze(1),
            size=target_shp,
            mode="bilinear",
            align_corners=False,
        )
        layer_maps.append(score.squeeze(1))
    return layer_maps


def fuse_layer_maps(layer_maps, aggregation="mean", top_k=3):
    """Fuse selected layer maps.

    ``mean`` preserves the historical behaviour. ``top-k`` implements the
    proposed per-pixel top-k aggregation over z-score comparable layers.
    """
    if not layer_maps:
        raise ValueError("cannot fuse an empty layer map list")
    if aggregation == "mean":
        return sum(layer_maps) / len(layer_maps)
    if aggregation == "top-k":
        stacked = torch.stack(layer_maps, dim=0)
        k = min(max(int(top_k), 1), stacked.shape[0])
        return torch.topk(stacked, k=k, dim=0).values.mean(dim=0)
    raise ValueError(f"unknown layer aggregation mode: {aggregation!r}")


def _apply_zscore_to_layer_maps(layer_maps, selected_layers, zscore_stats, stats_key):
    if stats_key is None:
        raise ValueError("stats_key is required when zscore_stats is enabled")
    method_stats = zscore_stats.get(stats_key)
    if method_stats is None:
        raise KeyError(f"missing z-score stats for {stats_key!r}")

    out = []
    for layer_map, layer in zip(layer_maps, selected_layers):
        layer_stats = method_stats[str(int(layer))]
        mu = float(layer_stats["mean"])
        sigma = max(float(layer_stats["std"]), 1e-6)
        out.append((layer_map - mu) / sigma)
    return out


# ---------------------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------------------

def plot_training_curves(logs_dir, out_path):
    import glob
    files = sorted(glob.glob(str(Path(logs_dir) / "*.pkl")),
                   key=lambda x: int(Path(x).stem.split(".")[0]))
    if not files:
        print("[diagnose] No training logs found — skipping curve.")
        return

    eps, losses, pos_sims, hn_sims = [], [], [], []
    for f in files:
        ep = int(Path(f).stem.split(".")[0])
        with open(f, "rb") as fd:
            d = pickle.load(fd)
        # Support both train_mydata (key="margin") and train_m2ad (key="contrastive")
        c = d.get("margin") or d.get("contrastive") or {}
        eps.append(ep)
        losses.append(d.get("loss", 0))
        pos_sims.append(c.get("pos_sim", 0))
        hn_sims.append(c.get("hard_neg_sim", 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eps, pos_sims, "g-", linewidth=1.5, label="pos_sim (N1↔N2)")
    axes[0].plot(eps, hn_sims, "r-", linewidth=1.5, label="hard_neg_sim (N1↔A1)")
    axes[0].set_title("Cosine Similarity")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eps, losses, "b-", linewidth=1.5)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[diagnose] Training curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Per-sample comparison visualization
# ---------------------------------------------------------------------------

def colorize_minmax(m, vmin=None, vmax=None):
    """Normalize to [0,1] using vmin/vmax (per-image min-max by default),
    then apply turbo colormap."""
    m = m.astype(np.float32)
    if vmin is None:
        vmin = m.min()
    if vmax is None:
        vmax = m.max()
    m = np.clip((m - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    cmap = plt.get_cmap("turbo")
    return (cmap(m)[..., :3] * 255).astype(np.uint8), m


def _render_sample_compare(
    out_path, ref_np, qry_np, gt_bin, piad_path, sc_b, sc_c, title,
    sc_b_vrange=None, sc_c_vrange=None,
):
    """Save a 2×3 comparison PNG for a single testdata sample.

    Layout:
        Row 1:  Ref  |  Query  |  GT mask
        Row 2:  PIAD heatmap.png  |  DINO baseline  |  Contrastive (trained)

    ``sc_b_vrange`` / ``sc_c_vrange`` are ``(vmin, vmax)`` tuples used for
    colormap normalisation. When the caller passes a category-level shared
    range, every sample of that category × method shows directly comparable
    colours. When ``None``, falls back to per-image min-max.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(ref_np); axes[0, 0].set_title("Ref")
    axes[0, 1].imshow(qry_np); axes[0, 1].set_title("Query")
    axes[0, 2].imshow(gt_bin * 255, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("GT mask")

    # PIAD heatmap is already a colorized RGB image — display as-is.
    if piad_path is not None and Path(piad_path).exists():
        piad_img = np.array(Image.open(piad_path).convert("RGB"))
        axes[1, 0].imshow(piad_img); axes[1, 0].set_title("PIAD heatmap")
    else:
        axes[1, 0].set_title("PIAD heatmap (missing)")

    def _title_for(name, sc, vrange):
        if vrange is not None:
            return (
                f"{name}\n"
                f"sample max={sc.max():.3f} (shared {vrange[0]:.2f}–{vrange[1]:.2f})"
            )
        return f"{name}\n[{sc.min():.3f}, {sc.max():.3f}]"

    if sc_b_vrange is not None:
        heat_b, _ = colorize_minmax(sc_b, vmin=sc_b_vrange[0], vmax=sc_b_vrange[1])
    else:
        heat_b, _ = colorize_minmax(sc_b)
    axes[1, 1].imshow(heat_b)
    axes[1, 1].set_title(_title_for("DINO baseline (untrained)", sc_b, sc_b_vrange))

    if sc_c_vrange is not None:
        heat_c, _ = colorize_minmax(sc_c, vmin=sc_c_vrange[0], vmax=sc_c_vrange[1])
    else:
        heat_c, _ = colorize_minmax(sc_c)
    axes[1, 2].imshow(heat_c)
    axes[1, 2].set_title(_title_for("Contrastive (trained)", sc_c, sc_c_vrange))

    for a in axes.ravel():
        a.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close()


def compute_auroc(gt_binary, score_map):
    try:
        from sklearn.metrics import roc_auc_score
        gt_flat = gt_binary.ravel().astype(int)
        sc_flat = score_map.ravel()
        if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
            return None
        return float(roc_auc_score(gt_flat, sc_flat))
    except ImportError:
        return None


def _get_defect_thresh(checkpoint_args, fallback=0.3):
    """Look up ``patch-defect-thresh`` from the saved training config.

    Different training entry points store loss params under different keys:
      - ``train_mydata``, ``train_m2ad_pair_mydata``: ``margin-loss``
      - ``train_m2ad`` (InfoNCE):                     ``contrastive-loss``

    Returns the first match (margin-loss preferred), else ``fallback``.
    """
    for section in ("margin-loss", "contrastive-loss"):
        cfg = checkpoint_args.get(section) or {}
        if "patch-defect-thresh" in cfg:
            return float(cfg["patch-defect-thresh"])
    return float(fallback)


def _resolve_use_layers(user_layers, trained_layers):
    """Validate requested inference layers and choose the fusion mode."""
    trained_layers = [int(layer) for layer in trained_layers]
    if user_layers is None:
        return list(trained_layers), "mean"

    tokens = [str(layer) for layer in user_layers]
    if not tokens:
        return list(trained_layers), "mean"
    if len(tokens) == 1 and tokens[0].lower() == "top-k":
        return list(trained_layers), "top-k"
    if any(token.lower() == "top-k" for token in tokens):
        raise SystemExit("--use-layers top-k must be used by itself.")

    try:
        selected_layers = [int(layer) for layer in tokens]
    except ValueError as exc:
        raise SystemExit(
            "--use-layers expects integer layer ids, or exactly: --use-layers top-k"
        ) from exc

    if len(set(selected_layers)) != len(selected_layers):
        raise SystemExit(f"--use-layers contains duplicates: {selected_layers}")

    allowed_layers = set(trained_layers)
    invalid_layers = [layer for layer in selected_layers if layer not in allowed_layers]
    if invalid_layers:
        raise SystemExit(
            f"--use-layers contains invalid layers {invalid_layers}. "
            f"Allowed layers from checkpoint are {trained_layers}."
        )
    return selected_layers, "mean"


# ---------------------------------------------------------------------------
# Quantitative evaluation on testdata
# ---------------------------------------------------------------------------
#
# testdata layout:
#   testdata/{Category}/{Category}_{N}/{ref.png, query.png, gt.png}
#
# - Each sample has its own (ref, query) pair plus a binary-ish GT mask
#   (gt > 0 → defective pixel).
# - Image size in testdata is 400×400; model expects ``figsize`` (e.g. 512×512),
#   so we resize before inference and resize the score map back to gt size
#   before computing AUROC against gt.
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_image_for_inference(path, figsize):
    """RGB → resize to ``figsize`` (H, W) → tensor → ImageNet-normalised."""
    img = Image.open(path).convert("RGB")
    img = img.resize((figsize[1], figsize[0]), Image.BILINEAR)
    t = tvff.to_tensor(img)
    t = tvff.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD)
    return t.unsqueeze(0)  # (1, 3, H, W)


def _resize_score_to(score_2d, target_hw):
    """Bilinearly resize a (H, W) score map to ``target_hw``."""
    t = torch.from_numpy(score_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def _list_testdata_samples(testdata_root):
    """Yield ``(category, sample_name, sample_dir)`` for every sample directory
    that contains all three of ref.png / query.png / gt.png.
    """
    root = Path(testdata_root)
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sample_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            ref_p = sample_dir / "ref.png"
            qry_p = sample_dir / "query.png"
            gt_p = sample_dir / "gt.png"
            if ref_p.exists() and qry_p.exists() and gt_p.exists():
                yield cat_dir.name, sample_dir.name, sample_dir


def _iter_mydata_normal_pairs(mydata_root):
    """Yield (ref.png, normal_query.png) pairs from the training mydata layout."""
    from robust_defect_detection.datasets.mydata_triplet import build_mydata_index

    index = build_mydata_index(mydata_root)
    for obj_name in sorted(index.keys()):
        entry = index[obj_name]
        ref = entry["ref"]
        for query in entry["normal_queries"]:
            yield obj_name, ref, query


def _new_running_stats(layers):
    return {
        str(int(layer)): {"sum": 0.0, "sum_sq": 0.0, "count": 0}
        for layer in layers
    }


def _update_running_stats(stats, layer, score_map):
    arr = score_map.detach().float()
    item = stats[str(int(layer))]
    item["sum"] += float(arr.sum().cpu())
    item["sum_sq"] += float((arr * arr).sum().cpu())
    item["count"] += int(arr.numel())


def _finalize_running_stats(stats):
    out = {}
    for layer, item in stats.items():
        count = max(int(item["count"]), 1)
        mean = item["sum"] / count
        var = max(item["sum_sq"] / count - mean * mean, 0.0)
        out[layer] = {
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "count": int(item["count"]),
        }
    return out


@torch.no_grad()
def estimate_zscore_stats(
    mydata_root,
    contrastive_encode,
    baseline_encode,
    trained_layers,
    target_shp,
    figsize,
    device,
    defect_thresh=0.3,
    max_pairs=None,
):
    """Estimate per-layer score mean/std from normal training ref-query pairs."""
    pairs = list(_iter_mydata_normal_pairs(mydata_root))
    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]
    if not pairs:
        raise SystemExit(f"No normal ref-query pairs found for z-score under {mydata_root}")

    raw_stats = _new_running_stats(trained_layers)
    trained_stats = _new_running_stats(trained_layers)
    print(f"[diagnose] Estimating z-score stats from {len(pairs)} normal training pairs")

    for obj_name, ref_path, query_path in pairs:
        ref_t = _load_image_for_inference(ref_path, figsize).to(device)
        qry_t = _load_image_for_inference(query_path, figsize).to(device)

        raw_maps = compute_layer_score_maps(
            baseline_encode,
            ref_t,
            qry_t,
            target_shp,
            defect_thresh=defect_thresh,
        )
        trained_maps = compute_layer_score_maps(
            contrastive_encode,
            ref_t,
            qry_t,
            target_shp,
            defect_thresh=defect_thresh,
        )
        for layer, raw_map, trained_map in zip(trained_layers, raw_maps, trained_maps):
            _update_running_stats(raw_stats, layer, raw_map)
            _update_running_stats(trained_stats, layer, trained_map)

    return {
        "source": str(mydata_root),
        "num_normal_pairs": len(pairs),
        "raw_dino": _finalize_running_stats(raw_stats),
        "trained": _finalize_running_stats(trained_stats),
    }


def load_or_estimate_zscore_stats(
    cache_path,
    mydata_root,
    contrastive_encode,
    baseline_encode,
    trained_layers,
    target_shp,
    figsize,
    device,
    defect_thresh=0.3,
    max_pairs=None,
):
    import json

    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[diagnose] Loading z-score stats → {cache_path}")
        with open(cache_path, "r") as fd:
            return json.load(fd)

    stats = estimate_zscore_stats(
        mydata_root=mydata_root,
        contrastive_encode=contrastive_encode,
        baseline_encode=baseline_encode,
        trained_layers=trained_layers,
        target_shp=target_shp,
        figsize=figsize,
        device=device,
        defect_thresh=defect_thresh,
        max_pairs=max_pairs,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fd:
        json.dump(stats, fd, indent=2)
    print(f"[diagnose] Saved z-score stats → {cache_path}")
    return stats


def _aggregate_auroc(gt_concat, score_concat):
    """AUROC over a concatenated 1-D pair, returning None if degenerate."""
    if gt_concat.size == 0:
        return None
    pos = int(gt_concat.sum())
    if pos == 0 or pos == len(gt_concat):
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(gt_concat, score_concat))
    except ImportError:
        return None


def evaluate_testdata(
    testdata_root,
    contrastive_encode,
    baseline_encode,
    target_shp,
    figsize,
    inference_cfg,
    device,
    out_dir,
    defect_thresh=0.3,
    trained_layers=None,
    use_layers=None,
    layer_aggregation="mean",
    top_k=3,
    zscore_stats=None,
):
    """Walk ``testdata`` and report pixel-level + image-level AUROC for both
    the trained (contrastive) and untrained (baseline) DINO features.

    Per-pixel AUROC is computed by concatenating every pixel score and gt
    label across all samples in a category; per-image AUROC uses
    ``max(score_map)`` as the image-level score and ``any(gt > 0)`` as the
    image-level label.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_root = out_dir / "visualizations"
    viz_root.mkdir(parents=True, exist_ok=True)

    # Group samples by category
    per_cat = {}
    for cat, name, sdir in _list_testdata_samples(testdata_root):
        per_cat.setdefault(cat, []).append((name, sdir))

    if not per_cat:
        print(f"[diagnose] testdata is empty under {testdata_root} — skipping.")
        return

    print(f"[diagnose] Quantitative eval on {sum(len(v) for v in per_cat.values())} "
          f"samples across {len(per_cat)} categories")

    # Per-sample rows + per-category aggregates
    rows = []                       # one per sample
    per_cat_summary = []            # one per category

    # For overall (micro) AUROC we accumulate across categories
    all_pixel_gt = []
    all_pixel_c, all_pixel_b = [], []
    all_img_gt = []
    all_img_c_max, all_img_b_max = [], []

    smooth_kernel = inference_cfg.get("gaussian-kernel", 5)
    smooth_sigma = inference_cfg.get("gaussian-sigma", 1.0)
    figsize = tuple(figsize)
    target_shp = tuple(target_shp)

    for cat in sorted(per_cat.keys()):
        samples = per_cat[cat]
        print(f"  [{cat}] {len(samples)} samples …")
        cat_viz_dir = viz_root / cat
        cat_viz_dir.mkdir(parents=True, exist_ok=True)

        cat_pixel_gt, cat_pixel_c, cat_pixel_b = [], [], []
        cat_img_gt, cat_img_c_max, cat_img_b_max = [], [], []

        # Pass 1: inference + AUROC accumulation. Stash score maps and
        # per-sample metadata so Pass 2 can render with a category-wide
        # per-method colour scale.
        deferred_render = []  # one dict per sample

        for name, sdir in samples:
            ref_t = _load_image_for_inference(sdir / "ref.png", figsize).to(device)
            qry_t = _load_image_for_inference(sdir / "query.png", figsize).to(device)

            gt_arr = np.array(Image.open(sdir / "gt.png").convert("L"))
            gt_bin = (gt_arr > 0).astype(np.uint8)
            gt_h, gt_w = gt_bin.shape

            with torch.no_grad():
                sc_c_full = compute_score_map(
                    contrastive_encode, ref_t, qry_t, target_shp, trained_layers,
                    {"gaussian-kernel": smooth_kernel, "gaussian-sigma": smooth_sigma},
                    defect_thresh=defect_thresh,
                    use_layers=use_layers,
                    aggregation=layer_aggregation,
                    top_k=top_k,
                    zscore_stats=zscore_stats,
                    stats_key="trained",
                )
                sc_b_full = compute_score_map(
                    baseline_encode, ref_t, qry_t, target_shp, trained_layers,
                    {"gaussian-kernel": smooth_kernel, "gaussian-sigma": smooth_sigma},
                    defect_thresh=defect_thresh,
                    use_layers=use_layers,
                    aggregation=layer_aggregation,
                    top_k=top_k,
                    zscore_stats=zscore_stats,
                    stats_key="raw_dino",
                )

            # Resize score maps from target_shp (e.g. 512) back to gt size
            sc_c = _resize_score_to(sc_c_full, (gt_h, gt_w))
            sc_b = _resize_score_to(sc_b_full, (gt_h, gt_w))

            img_label = int(gt_bin.any())
            cat_img_gt.append(img_label)
            cat_img_c_max.append(float(sc_c.max()))
            cat_img_b_max.append(float(sc_b.max()))

            cat_pixel_gt.append(gt_bin.ravel())
            cat_pixel_c.append(sc_c.ravel())
            cat_pixel_b.append(sc_b.ravel())

            # Per-sample row (only useful when sample is anomalous)
            sample_pix_auc_c = compute_auroc(gt_bin, sc_c) if img_label else None
            sample_pix_auc_b = compute_auroc(gt_bin, sc_b) if img_label else None
            rows.append({
                "category": cat,
                "sample": name,
                "anomalous": img_label,
                "pixel_auroc_contrastive": sample_pix_auc_c,
                "pixel_auroc_baseline": sample_pix_auc_b,
                "max_score_contrastive": float(sc_c.max()),
                "max_score_baseline": float(sc_b.max()),
            })

            deferred_render.append({
                "sdir": sdir,
                "name": name,
                "gt_bin": gt_bin,
                "sc_b": sc_b,
                "sc_c": sc_c,
                "img_label": img_label,
                "pix_auc_c": sample_pix_auc_c,
                "pix_auc_b": sample_pix_auc_b,
            })

        # ---- Per-category, per-method shared colour range ----
        # All samples of this category use a single (vmin, vmax) per method,
        # so heatmap colours are directly comparable across samples.
        cat_b_min = float(min(d["sc_b"].min() for d in deferred_render))
        cat_b_max = float(max(d["sc_b"].max() for d in deferred_render))
        cat_c_min = float(min(d["sc_c"].min() for d in deferred_render))
        cat_c_max = float(max(d["sc_c"].max() for d in deferred_render))

        # Pass 2: render per-sample figures using the shared vrange.
        for d in deferred_render:
            sdir = d["sdir"]
            ref_np = np.array(Image.open(sdir / "ref.png").convert("RGB"))
            qry_np = np.array(Image.open(sdir / "query.png").convert("RGB"))
            sample_title = f"{cat}/{d['name']}    anomalous={d['img_label']}"
            if d["pix_auc_c"] is not None:
                sample_title += (
                    f"    pixel-AUROC: ctv={d['pix_auc_c']:.3f}  "
                    f"base={d['pix_auc_b']:.3f}"
                )
            _render_sample_compare(
                out_path=cat_viz_dir / f"{d['name']}.png",
                ref_np=ref_np, qry_np=qry_np, gt_bin=d["gt_bin"],
                piad_path=sdir / "heatmap.png",
                sc_b=d["sc_b"], sc_c=d["sc_c"],
                title=sample_title,
                sc_b_vrange=(cat_b_min, cat_b_max),
                sc_c_vrange=(cat_c_min, cat_c_max),
            )

        # Concatenate within category
        cat_pixel_gt = np.concatenate(cat_pixel_gt)
        cat_pixel_c = np.concatenate(cat_pixel_c)
        cat_pixel_b = np.concatenate(cat_pixel_b)
        cat_img_gt = np.array(cat_img_gt, dtype=np.int64)
        cat_img_c_max = np.array(cat_img_c_max, dtype=np.float64)
        cat_img_b_max = np.array(cat_img_b_max, dtype=np.float64)

        pix_auc_c = _aggregate_auroc(cat_pixel_gt, cat_pixel_c)
        pix_auc_b = _aggregate_auroc(cat_pixel_gt, cat_pixel_b)
        img_auc_c = _aggregate_auroc(cat_img_gt, cat_img_c_max)
        img_auc_b = _aggregate_auroc(cat_img_gt, cat_img_b_max)

        per_cat_summary.append({
            "category": cat,
            "n_samples": len(samples),
            "n_anomalous": int(cat_img_gt.sum()),
            "pixel_auroc_contrastive": pix_auc_c,
            "pixel_auroc_baseline": pix_auc_b,
            "image_auroc_contrastive": img_auc_c,
            "image_auroc_baseline": img_auc_b,
            "delta_pixel_auroc": (pix_auc_c - pix_auc_b) if (pix_auc_c is not None and pix_auc_b is not None) else None,
            "delta_image_auroc": (img_auc_c - img_auc_b) if (img_auc_c is not None and img_auc_b is not None) else None,
        })

        all_pixel_gt.append(cat_pixel_gt)
        all_pixel_c.append(cat_pixel_c)
        all_pixel_b.append(cat_pixel_b)
        all_img_gt.append(cat_img_gt)
        all_img_c_max.append(cat_img_c_max)
        all_img_b_max.append(cat_img_b_max)

    # Overall (micro) AUROC: concatenate across categories
    all_pixel_gt = np.concatenate(all_pixel_gt)
    all_pixel_c = np.concatenate(all_pixel_c)
    all_pixel_b = np.concatenate(all_pixel_b)
    all_img_gt = np.concatenate(all_img_gt)
    all_img_c_max = np.concatenate(all_img_c_max)
    all_img_b_max = np.concatenate(all_img_b_max)

    overall_pix_c = _aggregate_auroc(all_pixel_gt, all_pixel_c)
    overall_pix_b = _aggregate_auroc(all_pixel_gt, all_pixel_b)
    overall_img_c = _aggregate_auroc(all_img_gt, all_img_c_max)
    overall_img_b = _aggregate_auroc(all_img_gt, all_img_b_max)

    # Macro: mean over categories (skip None)
    def _mean_or_none(xs, key):
        vals = [r[key] for r in xs if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    macro_pix_c = _mean_or_none(per_cat_summary, "pixel_auroc_contrastive")
    macro_pix_b = _mean_or_none(per_cat_summary, "pixel_auroc_baseline")
    macro_img_c = _mean_or_none(per_cat_summary, "image_auroc_contrastive")
    macro_img_b = _mean_or_none(per_cat_summary, "image_auroc_baseline")

    # ---- Save CSV ----
    # Column naming follows the convention of the existing testdata benchmark
    # CSV: ``img_roc_auc`` for image-level AUROC, ``per_pixel_rocauc`` for
    # pixel-level AUROC.
    import csv
    cat_csv = out_dir / "testdata_per_category.csv"
    with open(cat_csv, "w", newline="") as fd:
        w = csv.writer(fd)
        w.writerow([
            "category", "n_samples", "n_anomalous",
            "img_roc_auc_contrastive", "img_roc_auc_baseline", "img_roc_auc_delta",
            "per_pixel_rocauc_contrastive", "per_pixel_rocauc_baseline", "per_pixel_rocauc_delta",
        ])
        for r in per_cat_summary:
            w.writerow([
                r["category"], r["n_samples"], r["n_anomalous"],
                _fmt(r["image_auroc_contrastive"]), _fmt(r["image_auroc_baseline"]), _fmt(r["delta_image_auroc"]),
                _fmt(r["pixel_auroc_contrastive"]), _fmt(r["pixel_auroc_baseline"]), _fmt(r["delta_pixel_auroc"]),
            ])
        # Overall rows at the bottom
        w.writerow([])
        w.writerow(["MACRO_MEAN", "", "",
                    _fmt(macro_img_c), _fmt(macro_img_b),
                    _fmt(_safe_diff(macro_img_c, macro_img_b)),
                    _fmt(macro_pix_c), _fmt(macro_pix_b),
                    _fmt(_safe_diff(macro_pix_c, macro_pix_b))])
        w.writerow(["MICRO_OVERALL", len(all_img_gt), int(all_img_gt.sum()),
                    _fmt(overall_img_c), _fmt(overall_img_b),
                    _fmt(_safe_diff(overall_img_c, overall_img_b)),
                    _fmt(overall_pix_c), _fmt(overall_pix_b),
                    _fmt(_safe_diff(overall_pix_c, overall_pix_b))])

    sample_csv = out_dir / "testdata_per_sample.csv"
    with open(sample_csv, "w", newline="") as fd:
        w = csv.writer(fd)
        w.writerow([
            "category", "sample", "anomalous",
            "per_pixel_rocauc_contrastive", "per_pixel_rocauc_baseline",
            "max_score_contrastive", "max_score_baseline",
        ])
        for r in rows:
            w.writerow([
                r["category"], r["sample"], r["anomalous"],
                _fmt(r["pixel_auroc_contrastive"]), _fmt(r["pixel_auroc_baseline"]),
                _fmt(r["max_score_contrastive"]), _fmt(r["max_score_baseline"]),
            ])

    # ---- Print summary table ----
    # Image-level AUROC (label = gt.any(), score = max(score_map)) is shown
    # first because it is the headline metric for anomaly detection;
    # pixel-level AUROC follows as the localization metric.
    print("\n=== testdata AUROC evaluation ===")
    header = f"{'Category':<14} {'N':>4} {'Anom':>5} | " \
             f"{'imgAUC ctv':>10} {'imgAUC base':>11} {'Δimg':>7} | " \
             f"{'pixAUC ctv':>10} {'pixAUC base':>11} {'Δpix':>7}"
    print(header)
    print("-" * len(header))
    for r in per_cat_summary:
        print(f"{r['category']:<14} {r['n_samples']:>4} {r['n_anomalous']:>5} | "
              f"{_fmt(r['image_auroc_contrastive']):>10} "
              f"{_fmt(r['image_auroc_baseline']):>11} "
              f"{_fmt(r['delta_image_auroc']):>7} | "
              f"{_fmt(r['pixel_auroc_contrastive']):>10} "
              f"{_fmt(r['pixel_auroc_baseline']):>11} "
              f"{_fmt(r['delta_pixel_auroc']):>7}")
    print("-" * len(header))
    print(f"{'MACRO_MEAN':<14} {'':>4} {'':>5} | "
          f"{_fmt(macro_img_c):>10} {_fmt(macro_img_b):>11} "
          f"{_fmt(_safe_diff(macro_img_c, macro_img_b)):>7} | "
          f"{_fmt(macro_pix_c):>10} {_fmt(macro_pix_b):>11} "
          f"{_fmt(_safe_diff(macro_pix_c, macro_pix_b)):>7}")
    print(f"{'MICRO_OVERALL':<14} {len(all_img_gt):>4} {int(all_img_gt.sum()):>5} | "
          f"{_fmt(overall_img_c):>10} {_fmt(overall_img_b):>11} "
          f"{_fmt(_safe_diff(overall_img_c, overall_img_b)):>7} | "
          f"{_fmt(overall_pix_c):>10} {_fmt(overall_pix_b):>11} "
          f"{_fmt(_safe_diff(overall_pix_c, overall_pix_b)):>7}")

    # Headline AUROC (macro-mean across categories)
    print("\n" + "=" * 60)
    print("  AUROC headline (macro-mean across categories)")
    print("-" * 60)
    print(f"  image-level AUROC :  contrastive = {_fmt(macro_img_c)}    baseline = {_fmt(macro_img_b)}    Δ = {_fmt(_safe_diff(macro_img_c, macro_img_b))}")
    print(f"  pixel-level AUROC :  contrastive = {_fmt(macro_pix_c)}    baseline = {_fmt(macro_pix_b)}    Δ = {_fmt(_safe_diff(macro_pix_c, macro_pix_b))}")
    print("=" * 60)

    print(f"\n[diagnose] CSVs saved → {cat_csv}, {sample_csv}")
    print(f"[diagnose] Per-sample visualizations saved under {viz_root}/")


def _fmt(v):
    if v is None:
        return "  N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _safe_diff(a, b):
    if a is None or b is None:
        return None
    return a - b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--testdata-root", type=str, default="testdata",
                   help="testdata-style nested directory "
                        "({Category}/{Category}_{N}/{ref,query,gt,heatmap}.png).")
    p.add_argument("--output", type=str, default="outputs/diagnose")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use-layers", type=str, nargs="*", default=None,
                   help="Subset of checkpoint model.layers to fuse for "
                        "cosine-distance diagnosis. Default: all trained layers. "
                        "Use '--use-layers top-k' to enable per-pixel top-k "
                        "aggregation over all trained layers.")
    p.add_argument("--top-k", type=int, default=3,
                   help="k for '--use-layers top-k' aggregation. Default: 3.")
    p.add_argument("--Z-score", dest="z_score", action="store_true",
                   help="Enable per-layer z-score normalization estimated from "
                        "normal training ref-query pairs.")
    p.add_argument("--zscore-cache", type=str, default=None,
                   help="Optional path for z-score stats JSON. Default: "
                        "<output>/zscore_stats.json.")
    p.add_argument("--zscore-max-pairs", type=int, default=None,
                   help="Optional cap on normal training pairs used to estimate "
                        "z-score stats.")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    out_dir = Path(utils.resolve_path(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # ---- Load the trained model ----
    print("[diagnose] Loading checkpoint …")
    trained_model, checkpoint_data = models.load_checkpoint_model(
        checkpoint_path, device=device, verbose=False)
    trained_model.eval()

    model_name = checkpoint_data["args"]["model"].get("name", "")
    print(f"[diagnose] checkpoint model.name = {model_name!r}")

    # ---- Build baseline (raw DINOv3, no trained projector weights) ----
    baseline_args = checkpoint_data["args"]["model"]
    print("[diagnose] Building baseline model (raw DINOv3, untrained projector) …")
    baseline_model = models.get_model(**baseline_args)
    baseline_model = models.wrap_model_for_gpus(baseline_model, device=device)
    baseline_model.eval()

    inference_cfg = checkpoint_data["args"].get("inference", {})
    figsize = checkpoint_data["args"]["dataset"]["figsize"]
    target_shp = (
        int(checkpoint_data["args"]["model"]["target-shp-row"]),
        int(checkpoint_data["args"]["model"]["target-shp-col"]),
    )
    defect_thresh = _get_defect_thresh(checkpoint_data["args"], fallback=0.3)
    print(f"[diagnose] defect_thresh from checkpoint config: {defect_thresh}")

    trained_layers = [int(layer) for layer in baseline_args["layers"]]
    use_layers, layer_aggregation = _resolve_use_layers(args.use_layers, trained_layers)
    print(f"[diagnose] trained layers: {trained_layers}")
    print(f"[diagnose] using layers  : {use_layers}")
    print(f"[diagnose] layer fusion  : {layer_aggregation}")
    if layer_aggregation == "top-k":
        print(f"[diagnose] top-k         : {args.top_k}")

    # ---- Training curves ----
    logs_dir = Path(checkpoint_path).parent.parent / "logs"
    if not logs_dir.exists():
        logs_dir = Path(checkpoint_path).parent / "logs"
    plot_training_curves(logs_dir, str(out_dir / "training_curves.png"))

    # ---- Inference encode functions ----
    # callable(ref, query) → (ref_feats_per_layer, query_feats_per_layer)
    if isinstance(trained_model, torch.nn.DataParallel):
        c_encode = trained_model.module.encode_pair
    else:
        c_encode = trained_model.encode_pair

    # Baseline branch: always raw DINOv3 cosine distance
    if isinstance(baseline_model, torch.nn.DataParallel):
        b_encode = baseline_model.module.backbone
    else:
        b_encode = baseline_model.backbone

    zscore_stats = None
    if args.z_score:
        dataset_root = utils.resolve_path(checkpoint_data["args"]["dataset"]["root"])
        zscore_cache = (
            Path(utils.resolve_path(args.zscore_cache))
            if args.zscore_cache is not None
            else out_dir / "zscore_stats.json"
        )
        zscore_stats = load_or_estimate_zscore_stats(
            cache_path=zscore_cache,
            mydata_root=dataset_root,
            contrastive_encode=c_encode,
            baseline_encode=b_encode,
            trained_layers=trained_layers,
            target_shp=target_shp,
            figsize=figsize,
            device=device,
            defect_thresh=defect_thresh,
            max_pairs=args.zscore_max_pairs,
        )

    # ---- AUROC evaluation + per-sample visualisation on testdata ----
    testdata_root = utils.resolve_path(args.testdata_root)
    if not Path(testdata_root).exists():
        print(f"[diagnose] testdata-root does not exist: {testdata_root}")
        return

    evaluate_testdata(
        testdata_root=str(testdata_root),
        contrastive_encode=c_encode,
        baseline_encode=b_encode,
        target_shp=target_shp,
        figsize=figsize,
        inference_cfg=inference_cfg,
        device=device,
        out_dir=out_dir,
        defect_thresh=defect_thresh,
        trained_layers=trained_layers,
        use_layers=use_layers,
        layer_aggregation=layer_aggregation,
        top_k=args.top_k,
        zscore_stats=zscore_stats,
    )

    print(f"\n[diagnose] Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
