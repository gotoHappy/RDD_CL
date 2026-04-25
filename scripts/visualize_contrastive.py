import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tvff

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.datasets as datasets
import robust_defect_detection.models as models
from robust_defect_detection import utils


def get_default_root(name):
    return utils.resolve_path(name)


def build_blend_image(query_img, heatmap_rgb, query_alpha=0.7, heatmap_alpha=0.3):
    query_img = np.asarray(query_img, dtype=np.float32)
    heatmap_rgb = np.asarray(heatmap_rgb, dtype=np.float32)
    if query_img.max() > 1:
        query_img = query_img / 255.0
    if heatmap_rgb.max() > 1:
        heatmap_rgb = heatmap_rgb / 255.0
    return np.clip(query_alpha * query_img + heatmap_alpha * heatmap_rgb, 0.0, 1.0)


def colorize_score_map(score_map):
    score_map = np.asarray(score_map, dtype=np.float32)
    score_map = score_map - score_map.min()
    score_map = score_map / max(score_map.max(), 1e-8)
    cmap = plt.get_cmap("turbo")
    return cmap(score_map)[..., :3], score_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset", type=str, choices=["mydata", "mytestdata"], default="mydata")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/visualize_contrastive")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-layers", type=int, nargs="*", default=None)
    parser.add_argument("--layer-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def resolve_inference_layers(checkpoint_data, use_layers, layer_weights):
    trained_layers = [int(layer) for layer in checkpoint_data["args"]["model"]["layers"]]
    default_weights = checkpoint_data["args"].get("inference", {}).get("layer-weights")
    allowed_layers = set(trained_layers)

    if use_layers is None:
        selected_layers = list(trained_layers)
    else:
        selected_layers = [int(layer) for layer in use_layers]
        invalid_layers = [layer for layer in selected_layers if layer not in allowed_layers]
        if invalid_layers:
            raise ValueError(
                f"--use-layers contains invalid layers {invalid_layers}. "
                f"Allowed layers from checkpoint are {trained_layers}."
            )

    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError(f"--use-layers contains duplicates: {selected_layers}")

    if layer_weights is None:
        if default_weights is not None and len(default_weights) == len(trained_layers):
            weight_by_layer = {
                int(layer): float(weight)
                for layer, weight in zip(trained_layers, default_weights)
            }
            selected_weights = [weight_by_layer[layer] for layer in selected_layers]
        else:
            selected_weights = [1.0] * len(selected_layers)
    else:
        selected_weights = [float(weight) for weight in layer_weights]

    if len(selected_layers) != len(selected_weights):
        raise ValueError(
            f"--use-layers and --layer-weights must have the same length, "
            f"got {len(selected_layers)} and {len(selected_weights)}."
        )

    return trained_layers, selected_layers, selected_weights


def fuse_selected_layers(layer_maps, trained_layers, selected_layers, selected_weights):
    layer_map_by_id = {int(layer): layer_map for layer, layer_map in zip(trained_layers, layer_maps)}
    selected_maps = [layer_map_by_id[layer] for layer in selected_layers]
    weights = torch.tensor(
        selected_weights,
        dtype=selected_maps[0].dtype,
        device=selected_maps[0].device,
    )
    weights = weights / weights.sum().clamp_min(1e-8)

    fused_map = torch.zeros_like(selected_maps[0])
    for weight, layer_map in zip(weights, selected_maps):
        fused_map = fused_map + weight * layer_map
    return fused_map


def main():
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    dataset_root = os.path.abspath(args.dataset_root or get_default_root(args.dataset))
    output_dir = Path(utils.resolve_path(args.output))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    gpu_ids = args.gpu_ids
    if gpu_ids is None and str(device).startswith("cuda"):
        gpu_ids = [0]
    if str(device).startswith("cuda") and gpu_ids is not None and len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"

    model, checkpoint_data = models.load_checkpoint_model(
        checkpoint,
        device=device,
        gpu_ids=gpu_ids,
        verbose=False,
    )
    inference_cfg = checkpoint_data["args"].get("inference", {})
    trained_layers, selected_layers, selected_weights = resolve_inference_layers(
        checkpoint_data,
        args.use_layers,
        args.layer_weights,
    )
    figsize = checkpoint_data["args"]["dataset"]["figsize"]
    target_shp = (
        int(checkpoint_data["args"]["model"]["target-shp-row"]),
        int(checkpoint_data["args"]["model"]["target-shp-col"]),
    )

    if args.dataset == "mydata":
        base_dataset = datasets.get_dataset("mydata", root=dataset_root, mode="eval")
        loader = datasets.get_eval_loader(root=dataset_root, batch_size=1, num_workers=1, figsize=figsize)
    else:
        base_dataset = datasets.get_dataset("mytestdata", root=dataset_root)
        loader = datasets.get_inference_loader(root=dataset_root, batch_size=1, num_workers=1, figsize=figsize)

    if isinstance(model, torch.nn.DataParallel):
        encode_pair = model.module.encode_pair
    else:
        encode_pair = model.encode_pair

    for index, (ref_img, query_img, gt_mask) in enumerate(loader):
        if args.limit > 0 and index >= args.limit:
            break
        name = base_dataset.filenames[index].replace("/", "__")
        ref_np = ref_img.squeeze(0).permute(1, 2, 0).numpy()
        query_np = query_img.squeeze(0).permute(1, 2, 0).numpy()
        gt_np = gt_mask.squeeze(0).numpy()

        ref_img = ref_img.to(device)
        query_img = query_img.to(device)
        with torch.no_grad():
            ref_features, query_features = encode_pair(ref_img, query_img)
            layer_maps = []
            for ref_feature, query_feature in zip(ref_features, query_features):
                score = 1.0 - torch.nn.functional.cosine_similarity(ref_feature, query_feature, dim=1)
                score = score.unsqueeze(1)
                score = torch.nn.functional.interpolate(
                    score,
                    size=target_shp,
                    mode="bilinear",
                    align_corners=False,
                )
                layer_maps.append(score.squeeze(1))

            fused_map = fuse_selected_layers(
                layer_maps=layer_maps,
                trained_layers=trained_layers,
                selected_layers=selected_layers,
                selected_weights=selected_weights,
            )
            smooth_kernel = inference_cfg.get("gaussian-kernel", 5)
            if smooth_kernel and smooth_kernel > 1:
                smooth_sigma = inference_cfg.get("gaussian-sigma", 1.0)
                fused_map = tvff.gaussian_blur(
                    fused_map.unsqueeze(1),
                    kernel_size=[smooth_kernel, smooth_kernel],
                    sigma=[smooth_sigma, smooth_sigma],
                ).squeeze(1)
        score_map = fused_map.squeeze(0).detach().cpu().numpy()
        heatmap_rgb, score_norm = colorize_score_map(score_map)
        pred_mask = score_norm >= args.threshold
        blend = build_blend_image(query_np, heatmap_rgb)

        num_cols = 6 if args.dataset == "mydata" else 5
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
        axes[0].imshow(np.clip(ref_np, 0, 1))
        axes[0].set_title("Reference")
        axes[1].imshow(np.clip(query_np, 0, 1))
        axes[1].set_title("Query")
        axes[2].imshow(heatmap_rgb)
        axes[2].set_title("Heatmap")
        axes[3].imshow(np.clip(blend, 0, 1))
        axes[3].set_title("Blend")
        axes[4].imshow(pred_mask, cmap="gray")
        axes[4].set_title(f"Pred Mask >= {args.threshold:.2f}")

        if args.dataset == "mydata":
            axes[5].imshow(gt_np, cmap="gray")
            axes[5].set_title("GT Mask")

        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"layers={selected_layers} weights={selected_weights}",
            fontsize=10,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
