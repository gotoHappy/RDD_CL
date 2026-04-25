"""
Diagnostic script for comparing contrastive vs baseline inference.

Outputs:
  1. Training curve plot (all epochs)
  2. Per-sample score comparison:
       - raw scores histogram (no normalization)
       - shared-scale side-by-side heatmaps
       - projector delta map (contrastive − baseline)
  3. Per-sample statistics table printed to console

Usage:
    python scripts/diagnose_m2ad.py <checkpoint.pth> \
        --dataset-root mytestdata \
        --output outputs/diagnose \
        [--gt-mask-root <path>]      # optional: directory with binary GT masks

If gt-mask-root is provided, per-sample AUROC (pixel-level) is computed.
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

import robust_defect_detection.datasets as datasets
import robust_defect_detection.models as models
from robust_defect_detection import utils


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_score_map(encode_fn, ref_img, query_img, target_shp, trained_layers, smooth_cfg):
    ref_features, query_features = encode_fn(ref_img, query_img)
    layer_maps = []
    for rf, qf in zip(ref_features, query_features):
        score = 1.0 - F.cosine_similarity(rf, qf, dim=1)
        score = F.interpolate(score.unsqueeze(1), size=target_shp, mode="bilinear", align_corners=False)
        layer_maps.append(score.squeeze(1))
    fused = sum(layer_maps) / len(layer_maps)
    k = smooth_cfg.get("gaussian-kernel", 5)
    if k and k > 1:
        sig = smooth_cfg.get("gaussian-sigma", 1.0)
        fused = tvff.gaussian_blur(fused.unsqueeze(1), [k, k], [sig, sig]).squeeze(1)
    return fused.squeeze(0).detach().cpu().numpy()  # (H, W)


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
    """Normalize to [0,1] using global vmin/vmax, then apply turbo colormap."""
    m = m.astype(np.float32)
    if vmin is None:
        vmin = m.min()
    if vmax is None:
        vmax = m.max()
    m = np.clip((m - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    cmap = plt.get_cmap("turbo")
    return (cmap(m)[..., :3] * 255).astype(np.uint8), m


def compute_shared_vrange(*maps):
    combined = np.concatenate([m.ravel() for m in maps])
    return float(combined.min()), float(combined.max())


def try_load_gt(gt_root, name, target_shp):
    if gt_root is None:
        return None
    candidates = [
        Path(gt_root) / name / "mask.png",
        Path(gt_root) / name / "gt_mask.png",
        Path(gt_root) / (name + ".png"),
    ]
    for c in candidates:
        if c.exists():
            m = np.array(Image.open(c).convert("L").resize(
                (target_shp[1], target_shp[0]), Image.NEAREST), dtype=np.uint8)
            return (m > 0).astype(np.uint8)
    return None


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--dataset-root", type=str, default="mytestdata")
    p.add_argument("--output", type=str, default="outputs/diagnose")
    p.add_argument("--gt-mask-root", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    dataset_root = utils.resolve_path(args.dataset_root)
    out_dir = Path(utils.resolve_path(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_root = args.gt_mask_root

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # ---- Load models ----
    print("[diagnose] Loading checkpoint for contrastive model …")
    contrastive_model, checkpoint_data = models.load_checkpoint_model(
        checkpoint_path, device=device, verbose=False)
    contrastive_model.eval()

    print("[diagnose] Building baseline model (same backbone, no projector weights) …")
    baseline_model = models.get_model(**checkpoint_data["args"]["model"])
    baseline_model = models.wrap_model_for_gpus(baseline_model, device=device)
    baseline_model.eval()

    inference_cfg = checkpoint_data["args"].get("inference", {})
    trained_layers = [int(l) for l in checkpoint_data["args"]["model"]["layers"]]
    figsize = checkpoint_data["args"]["dataset"]["figsize"]
    target_shp = (
        int(checkpoint_data["args"]["model"]["target-shp-row"]),
        int(checkpoint_data["args"]["model"]["target-shp-col"]),
    )

    # ---- Training curves ----
    logs_dir = Path(checkpoint_path).parent.parent / "logs"
    if not logs_dir.exists():
        # try sibling of checkpoint
        logs_dir = Path(checkpoint_path).parent / "logs"
    plot_training_curves(logs_dir, str(out_dir / "training_curves.png"))

    # ---- Inference encode functions ----
    if isinstance(contrastive_model, torch.nn.DataParallel):
        c_encode = contrastive_model.module.encode_pair
        b_encode = baseline_model.module.backbone
    else:
        c_encode = contrastive_model.encode_pair
        b_encode = baseline_model.backbone

    # ---- Dataloader ----
    dataset = datasets.get_dataset("mytestdata", root=str(dataset_root))
    loader = datasets.get_inference_loader(
        root=str(dataset_root), batch_size=1, num_workers=0, figsize=figsize)

    all_stats = []

    for index, (ref_img, query_img, _) in enumerate(loader):
        name = dataset.filenames[index]
        ref_t = ref_img.to(device)
        qry_t = query_img.to(device)

        with torch.no_grad():
            score_c = compute_score_map(c_encode, ref_t, qry_t, target_shp, trained_layers, inference_cfg)
            score_b = compute_score_map(b_encode, ref_t, qry_t, target_shp, trained_layers, inference_cfg)

        # Shared color scale
        vmin, vmax = compute_shared_vrange(score_c, score_b)
        heat_c, norm_c = colorize_minmax(score_c, vmin, vmax)
        heat_b, norm_b = colorize_minmax(score_b, vmin, vmax)

        # Also per-image normalized (what infer_*.py shows)
        heat_c_local, _ = colorize_minmax(score_c)
        heat_b_local, _ = colorize_minmax(score_b)

        # Delta map (contrastive − baseline, signed)
        delta = score_c - score_b
        delta_norm = (delta - delta.min()) / max(delta.max() - delta.min(), 1e-8)
        heat_delta = (plt.get_cmap("RdBu_r")(delta_norm)[..., :3] * 255).astype(np.uint8)

        # Query / Ref images
        qry_np = (np.clip(query_img.squeeze(0).permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
        ref_np = (np.clip(ref_img.squeeze(0).permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)

        # GT mask (optional)
        gt = try_load_gt(gt_root, name, target_shp)
        auroc_c = compute_auroc(gt, score_c) if gt is not None else None
        auroc_b = compute_auroc(gt, score_b) if gt is not None else None

        # Save comparison image
        sample_dir = out_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        # Row 1: shared scale
        axes[0, 0].imshow(qry_np); axes[0, 0].set_title("Query")
        axes[0, 1].imshow(heat_b); axes[0, 1].set_title(f"Baseline (shared scale)\n[{vmin:.3f}, {vmax:.3f}]")
        axes[0, 2].imshow(heat_c); axes[0, 2].set_title(f"Contrastive (shared scale)\n[{vmin:.3f}, {vmax:.3f}]")
        axes[0, 3].imshow(heat_delta); axes[0, 3].set_title("Δ (contrastive − baseline)\nred=contrastive higher")
        # Row 2: per-image normalized (what infer_*.py would show)
        axes[1, 0].imshow(ref_np); axes[1, 0].set_title("Ref")
        axes[1, 1].imshow(heat_b_local); axes[1, 1].set_title("Baseline (per-img norm)")
        axes[1, 2].imshow(heat_c_local); axes[1, 2].set_title("Contrastive (per-img norm)")

        # Score histograms
        ax = axes[1, 3]
        ax.hist(score_b.ravel(), bins=60, alpha=0.6, label="baseline", color="steelblue", density=True)
        ax.hist(score_c.ravel(), bins=60, alpha=0.6, label="contrastive", color="tomato", density=True)
        if gt is not None:
            # Highlight GT defect region scores
            ax.hist(score_b[gt > 0], bins=30, alpha=0.4, label="baseline@GT", color="navy", density=True)
            ax.hist(score_c[gt > 0], bins=30, alpha=0.4, label="contrastive@GT", color="darkred", density=True)
        ax.legend(fontsize=8); ax.set_title("Score distributions"); ax.set_xlabel("1 − cos_sim")

        for a in axes.ravel():
            a.axis("off") if a.get_images() else None

        auc_str = f"  AUROC  base={auroc_b:.3f}  ctv={auroc_c:.3f}" if auroc_c is not None else ""
        fig.suptitle(f"{name}{auc_str}", fontsize=12)
        plt.tight_layout()
        plt.savefig(str(sample_dir / "compare.png"), dpi=120)
        plt.close()

        stat = {
            "name": name,
            "baseline": {"mean": float(score_b.mean()), "std": float(score_b.std()),
                         "min": float(score_b.min()), "max": float(score_b.max())},
            "contrastive": {"mean": float(score_c.mean()), "std": float(score_c.std()),
                            "min": float(score_c.min()), "max": float(score_c.max())},
            "delta_abs_mean": float(np.abs(delta).mean()),
            "delta_max": float(delta.max()),
        }
        if auroc_c is not None:
            stat["auroc_baseline"] = auroc_b
            stat["auroc_contrastive"] = auroc_c
        all_stats.append(stat)
        print(f"  [{name}]  base: mean={stat['baseline']['mean']:.4f} std={stat['baseline']['std']:.4f}"
              f"  ctv: mean={stat['contrastive']['mean']:.4f} std={stat['contrastive']['std']:.4f}"
              f"  |Δ|_mean={stat['delta_abs_mean']:.4f}"
              + (f"  AUROC b={auroc_b:.3f}/c={auroc_c:.3f}" if auroc_c else ""))

    # Summary
    if all_stats:
        print("\n=== Summary ===")
        print(f"{'Name':<20} {'base_mean':>10} {'ctv_mean':>10} {'|Δ|_mean':>10} {'AUROC_b':>8} {'AUROC_c':>8}")
        for s in all_stats:
            auc_b = f"{s['auroc_baseline']:.3f}" if "auroc_baseline" in s else "  N/A"
            auc_c = f"{s['auroc_contrastive']:.3f}" if "auroc_contrastive" in s else "  N/A"
            print(f"{s['name']:<20} {s['baseline']['mean']:>10.4f} {s['contrastive']['mean']:>10.4f}"
                  f" {s['delta_abs_mean']:>10.4f} {auc_b:>8} {auc_c:>8}")

    print(f"\n[diagnose] Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
