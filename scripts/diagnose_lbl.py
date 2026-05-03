"""diagnose_lbl.py — layer-by-layer feature-difference visualisation.

For each testdata sample, render a figure with:

    Row 1:  Ref  |  Query  |  GT mask
    Row 2:  raw DINOv3 (untrained)  per-layer  1 − cos(ref, query)  score maps
    Row 3:  contrastive (trained)   per-layer  1 − cos(ref, query)  score maps

Per-row colour normalisation is shared across layers (so the absolute
score-magnitude difference between layers is visually comparable within a
method); the two rows have independent colour scales because raw DINO and
the trained projector live in very different magnitude regimes.

Layer selection
---------------
Pass ``--layers L1 L2 ...`` to visualise a subset of the layers the model
was trained on. The integers must come from
``checkpoint["args"]["model"]["layers"]``. With no flag, all trained layers
are visualised.

Only ``dino2 + contrast_learning`` checkpoints are supported.

Usage
-----
::

    python scripts/diagnose_lbl.py <ckpt.pth> \\
        --testdata-root testdata \\
        --output outputs/.../diagnose_lbl \\
        [--layers 9 10 11]
"""

import argparse
import os
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
# I/O helpers (kept identical to diagnose.py to avoid drift)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_image_for_inference(path, figsize):
    img = Image.open(path).convert("RGB")
    img = img.resize((figsize[1], figsize[0]), Image.BILINEAR)
    t = tvff.to_tensor(img)
    return tvff.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD).unsqueeze(0)


def _resize_to(score_2d, target_hw):
    t = torch.from_numpy(score_2d).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def _list_testdata_samples(root):
    root = Path(root)
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sd in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            if all((sd / n).exists() for n in ("ref.png", "query.png", "gt.png")):
                yield cat_dir.name, sd.name, sd


# ---------------------------------------------------------------------------
# Per-layer score computation
# ---------------------------------------------------------------------------

def compute_layer_scores(encode_fn, ref_t, qry_t, target_shp):
    """One score map per extractor layer.

    The score is the *raw* cosine distance ``1 − cos(ref, query)`` at the
    patch grid, bilinearly upsampled to ``target_shp``. We deliberately do
    NOT apply the gamma suppression that ``diagnose.py`` uses: this script
    is for inspecting the underlying feature differences as the network
    sees them, not for producing a polished detection score.
    """
    ref_feats, qry_feats = encode_fn(ref_t, qry_t)
    out = []
    for rf, qf in zip(ref_feats, qry_feats):
        s = 1.0 - F.cosine_similarity(rf, qf, dim=1)            # (B, h, w)
        s = F.interpolate(s.unsqueeze(1), size=target_shp,
                          mode="bilinear", align_corners=False)
        out.append(s.squeeze(0).squeeze(0).detach().cpu().numpy())
    return out


def colorize_minmax(m, vmin, vmax):
    m = m.astype(np.float32)
    m = np.clip((m - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    cmap = plt.get_cmap("turbo")
    return (cmap(m)[..., :3] * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-sample figure
# ---------------------------------------------------------------------------

def render_layer_compare(
    out_path,
    ref_np, qry_np, gt_bin,
    baseline_layer_scores,
    trained_layer_scores,
    layer_indices,
    title,
):
    N = len(layer_indices)
    W = max(3, N)
    fig = plt.figure(figsize=(3.2 * W, 10.5))
    gs = fig.add_gridspec(3, W)

    # Row 1: ref / query / gt — first three columns
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(ref_np); ax.set_title("Ref"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(qry_np); ax.set_title("Query"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(gt_bin * 255, cmap="gray", vmin=0, vmax=255)
    ax.set_title("GT mask"); ax.axis("off")
    for j in range(3, W):
        fig.add_subplot(gs[0, j]).axis("off")

    # Row 2 / 3 use independent shared colour ranges
    base_vmin = float(min(s.min() for s in baseline_layer_scores))
    base_vmax = float(max(s.max() for s in baseline_layer_scores))
    train_vmin = float(min(s.min() for s in trained_layer_scores))
    train_vmax = float(max(s.max() for s in trained_layer_scores))

    # Row 2: raw DINOv3 per-layer score
    for i, (li, sc) in enumerate(zip(layer_indices, baseline_layer_scores)):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(colorize_minmax(sc, base_vmin, base_vmax))
        ax.set_title(
            f"raw DINOv3  layer {li}\n"
            f"sample [{sc.min():.3f}, {sc.max():.3f}]\n"
            f"shared [{base_vmin:.3f}, {base_vmax:.3f}]",
            fontsize=9,
        )
        ax.axis("off")
    for j in range(N, W):
        fig.add_subplot(gs[1, j]).axis("off")

    # Row 3: contrastive (trained) per-layer score
    for i, (li, sc) in enumerate(zip(layer_indices, trained_layer_scores)):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(colorize_minmax(sc, train_vmin, train_vmax))
        ax.set_title(
            f"trained  layer {li}\n"
            f"sample [{sc.min():.3f}, {sc.max():.3f}]\n"
            f"shared [{train_vmin:.3f}, {train_vmax:.3f}]",
            fontsize=9,
        )
        ax.axis("off")
    for j in range(N, W):
        fig.add_subplot(gs[2, j]).axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--testdata-root", type=str, default="testdata")
    p.add_argument("--output", type=str, default="outputs/diagnose_lbl")
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="Layer indices to visualise (subset of "
                        "checkpoint['args']['model']['layers']). Default: "
                        "all trained layers, in their training order.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _resolve_layers(user_layers, trained_layers):
    if not user_layers:
        return list(trained_layers)
    # Dedup but preserve order of first occurrence
    seen = set()
    selected = []
    for l in user_layers:
        l = int(l)
        if l in seen:
            continue
        seen.add(l)
        selected.append(l)
    invalid = [l for l in selected if l not in trained_layers]
    if invalid:
        raise SystemExit(
            f"--layers {invalid} are not in the trained layer set {trained_layers}"
        )
    return selected


def main():
    args = parse_args()
    out_dir = Path(utils.resolve_path(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_root = out_dir / "visualizations"
    viz_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    print("[diagnose_lbl] loading checkpoint …")
    trained, ckpt_data = models.load_checkpoint_model(
        args.checkpoint, device=device, verbose=False)
    trained.eval()

    name = ckpt_data["args"]["model"].get("name", "")
    if name != "dino2 + contrast_learning":
        raise SystemExit(
            f"diagnose_lbl only supports 'dino2 + contrast_learning' "
            f"checkpoints (got {name!r})."
        )

    print("[diagnose_lbl] building baseline (raw DINOv3, untrained projector) …")
    baseline = models.get_model(**ckpt_data["args"]["model"])
    baseline = models.wrap_model_for_gpus(baseline, device=device)
    baseline.eval()

    trained_layers = [int(l) for l in ckpt_data["args"]["model"]["layers"]]
    selected = _resolve_layers(args.layers, trained_layers)
    print(f"[diagnose_lbl] trained layers: {trained_layers}")
    print(f"[diagnose_lbl] visualising  : {selected}")
    sel_indices = [trained_layers.index(l) for l in selected]

    figsize = ckpt_data["args"]["dataset"]["figsize"]
    target_shp = (
        int(ckpt_data["args"]["model"]["target-shp-row"]),
        int(ckpt_data["args"]["model"]["target-shp-col"]),
    )

    if isinstance(trained, torch.nn.DataParallel):
        c_encode = trained.module.encode_pair
        b_encode = baseline.module.backbone
    else:
        c_encode = trained.encode_pair
        b_encode = baseline.backbone

    testdata_root = utils.resolve_path(args.testdata_root)
    if not Path(testdata_root).exists():
        raise SystemExit(f"testdata-root not found: {testdata_root}")

    all_samples = list(_list_testdata_samples(testdata_root))
    print(f"[diagnose_lbl] processing {len(all_samples)} samples")

    for cat, sample_name, sdir in all_samples:
        cat_viz = viz_root / cat
        cat_viz.mkdir(parents=True, exist_ok=True)

        ref_t = _load_image_for_inference(sdir / "ref.png", figsize).to(device)
        qry_t = _load_image_for_inference(sdir / "query.png", figsize).to(device)

        gt_arr = np.array(Image.open(sdir / "gt.png").convert("L"))
        gt_bin = (gt_arr > 0).astype(np.uint8)
        gt_h, gt_w = gt_bin.shape

        with torch.no_grad():
            base_full = compute_layer_scores(b_encode, ref_t, qry_t, target_shp)
            trained_full = compute_layer_scores(c_encode, ref_t, qry_t, target_shp)

        # Pick requested layers and resize to gt resolution
        base_scores = [_resize_to(base_full[i], (gt_h, gt_w)) for i in sel_indices]
        trained_scores = [_resize_to(trained_full[i], (gt_h, gt_w)) for i in sel_indices]

        ref_np = np.array(Image.open(sdir / "ref.png").convert("RGB"))
        qry_np = np.array(Image.open(sdir / "query.png").convert("RGB"))

        title = f"{cat}/{sample_name}    anomalous={int(gt_bin.any())}    " \
                f"layers={selected}"

        render_layer_compare(
            out_path=cat_viz / f"{sample_name}.png",
            ref_np=ref_np, qry_np=qry_np, gt_bin=gt_bin,
            baseline_layer_scores=base_scores,
            trained_layer_scores=trained_scores,
            layer_indices=selected,
            title=title,
        )

    print(f"[diagnose_lbl] done. {len(all_samples)} figures saved under {viz_root}/")


if __name__ == "__main__":
    main()
