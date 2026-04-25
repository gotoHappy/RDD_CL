import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tvff
from torchvision.transforms import InterpolationMode

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.models as models
from robust_defect_detection import utils


def build_blend_image(query_img, heatmap_rgb, query_alpha=0.7, heatmap_alpha=0.3):
    query_img = np.asarray(query_img, dtype=np.float32)
    heatmap_rgb = np.asarray(heatmap_rgb, dtype=np.float32)
    if query_img.max() > 1:
        query_img = query_img / 255.0
    if heatmap_rgb.max() > 1:
        heatmap_rgb = heatmap_rgb / 255.0
    blend = np.clip(query_alpha * query_img + heatmap_alpha * heatmap_rgb, 0.0, 1.0)
    return (blend * 255).astype(np.uint8)


def colorize_score_map(score_map):
    score_map = np.asarray(score_map, dtype=np.float32)
    score_map = score_map - score_map.min()
    score_map = score_map / max(score_map.max(), 1e-8)
    cmap = plt.get_cmap("turbo")
    return (cmap(score_map)[..., :3] * 255).astype(np.uint8), score_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset-root", type=str, default="mytestdata")
    parser.add_argument("--output", type=str, default="outputs/infer_baseline_R")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-layers", type=int, nargs="*", default=None)
    parser.add_argument("--layer-weights", type=float, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
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


def build_baseline_model(checkpoint_data, device, gpu_ids):
    model = models.get_model(**checkpoint_data["args"]["model"])
    model = models.wrap_model_for_gpus(model, device=device, gpu_ids=gpu_ids)
    model.eval()
    return model


def load_resized_tensor(path, figsize):
    image = Image.open(path).convert("RGB")
    image = tvff.resize(image, figsize, interpolation=InterpolationMode.BILINEAR)
    return tvff.to_tensor(image)


def collect_reflective_samples(dataset_root):
    samples = []
    for obj_dir in sorted(Path(dataset_root).glob("obj_*")):
        ref_path = obj_dir / "ref.png"
        query_path = obj_dir / "query.png"
        rref_path = obj_dir / "Rref.png"
        rquery_path = obj_dir / "Rquery.png"
        if ref_path.exists() and query_path.exists() and rref_path.exists() and rquery_path.exists():
            samples.append(
                {
                    "name": obj_dir.name,
                    "ref_path": ref_path,
                    "query_path": query_path,
                    "rref_path": rref_path,
                    "rquery_path": rquery_path,
                }
            )
    return samples


def compute_multilayer_maps(encode_pair, ref_img, query_img, target_shp):
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
    return layer_maps


def multiply_layer_maps(rgb_layer_maps, r_layer_maps):
    if len(rgb_layer_maps) != len(r_layer_maps):
        raise ValueError(
            f"rgb_layer_maps and r_layer_maps must have the same length, "
            f"got {len(rgb_layer_maps)} and {len(r_layer_maps)}."
        )
    return [rgb_map * r_map for rgb_map, r_map in zip(rgb_layer_maps, r_layer_maps)]


def main():
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")

    dataset_root = utils.resolve_path(args.dataset_root)
    output_dir = utils.resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    gpu_ids = args.gpu_ids
    if gpu_ids is None and str(device).startswith("cuda"):
        gpu_ids = [0]
    if str(device).startswith("cuda") and gpu_ids is not None and len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"

    model = build_baseline_model(checkpoint_data, device=device, gpu_ids=gpu_ids)
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

    samples = collect_reflective_samples(dataset_root)
    if not samples:
        raise ValueError(f"no samples with ref/query/Rref/Rquery found under {dataset_root}")

    if isinstance(model, torch.nn.DataParallel):
        encode_pair = model.module.backbone
    else:
        encode_pair = model.backbone

    summary = []
    skipped = []
    for sample in samples:
        name = sample["name"]
        ref_tensor = load_resized_tensor(sample["ref_path"], figsize)
        query_tensor = load_resized_tensor(sample["query_path"], figsize)
        rref_tensor = load_resized_tensor(sample["rref_path"], figsize)
        rquery_tensor = load_resized_tensor(sample["rquery_path"], figsize)

        ref_np = ref_tensor.permute(1, 2, 0).numpy()
        query_np = query_tensor.permute(1, 2, 0).numpy()
        rref_np = rref_tensor.permute(1, 2, 0).numpy()
        rquery_np = rquery_tensor.permute(1, 2, 0).numpy()

        ref_img = ref_tensor.unsqueeze(0).to(device)
        query_img = query_tensor.unsqueeze(0).to(device)
        rref_img = rref_tensor.unsqueeze(0).to(device)
        rquery_img = rquery_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            rgb_layer_maps = compute_multilayer_maps(encode_pair, ref_img, query_img, target_shp)
            r_layer_maps = compute_multilayer_maps(encode_pair, rref_img, rquery_img, target_shp)
            total_layer_maps = multiply_layer_maps(rgb_layer_maps, r_layer_maps)

            rgb_fused_map = fuse_selected_layers(
                layer_maps=rgb_layer_maps,
                trained_layers=trained_layers,
                selected_layers=selected_layers,
                selected_weights=selected_weights,
            )
            r_fused_map = fuse_selected_layers(
                layer_maps=r_layer_maps,
                trained_layers=trained_layers,
                selected_layers=selected_layers,
                selected_weights=selected_weights,
            )
            total_map = fuse_selected_layers(
                layer_maps=total_layer_maps,
                trained_layers=trained_layers,
                selected_layers=selected_layers,
                selected_weights=selected_weights,
            )

            smooth_kernel = inference_cfg.get("gaussian-kernel", 5)
            if smooth_kernel and smooth_kernel > 1:
                smooth_sigma = inference_cfg.get("gaussian-sigma", 1.0)
                rgb_fused_map = tvff.gaussian_blur(
                    rgb_fused_map.unsqueeze(1),
                    kernel_size=[smooth_kernel, smooth_kernel],
                    sigma=[smooth_sigma, smooth_sigma],
                ).squeeze(1)
                r_fused_map = tvff.gaussian_blur(
                    r_fused_map.unsqueeze(1),
                    kernel_size=[smooth_kernel, smooth_kernel],
                    sigma=[smooth_sigma, smooth_sigma],
                ).squeeze(1)
                total_map = tvff.gaussian_blur(
                    total_map.unsqueeze(1),
                    kernel_size=[smooth_kernel, smooth_kernel],
                    sigma=[smooth_sigma, smooth_sigma],
                ).squeeze(1)

        rgb_score_map = rgb_fused_map.squeeze(0).detach().cpu().numpy()
        r_score_map = r_fused_map.squeeze(0).detach().cpu().numpy()
        total_score_map = total_map.squeeze(0).detach().cpu().numpy()

        rgb_heatmap_rgb, rgb_score_norm = colorize_score_map(rgb_score_map)
        r_heatmap_rgb, r_score_norm = colorize_score_map(r_score_map)
        total_heatmap_rgb, total_score_norm = colorize_score_map(total_score_map)

        binary_mask = (total_score_norm >= args.threshold).astype(np.uint8) * 255
        blend = build_blend_image(query_np, total_heatmap_rgb)

        sample_dir = output_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(ref_np, 0.0, 1.0) * 255).astype(np.uint8)).save(sample_dir / "ref.png")
        Image.fromarray((np.clip(query_np, 0.0, 1.0) * 255).astype(np.uint8)).save(sample_dir / "query.png")
        Image.fromarray((np.clip(rref_np, 0.0, 1.0) * 255).astype(np.uint8)).save(sample_dir / "Rref.png")
        Image.fromarray((np.clip(rquery_np, 0.0, 1.0) * 255).astype(np.uint8)).save(sample_dir / "Rquery.png")
        Image.fromarray(rgb_heatmap_rgb).save(sample_dir / "rgb_heatmap.png")
        Image.fromarray((rgb_score_norm * 255).astype(np.uint8)).save(sample_dir / "rgb_score_map.png")
        Image.fromarray(r_heatmap_rgb).save(sample_dir / "R_heatmap.png")
        Image.fromarray((r_score_norm * 255).astype(np.uint8)).save(sample_dir / "R_score_map.png")
        Image.fromarray(total_heatmap_rgb).save(sample_dir / "heatmap.png")
        Image.fromarray((total_score_norm * 255).astype(np.uint8)).save(sample_dir / "score_map.png")
        Image.fromarray(binary_mask).save(sample_dir / "pred_mask.png")
        Image.fromarray(blend).save(sample_dir / "blend.png")

        summary.append(
            {
                "sample": name,
                "threshold": args.threshold,
                "trained_layers": trained_layers,
                "used_layers": selected_layers,
                "used_layer_weights": selected_weights,
                "baseline": "Dinov3_rgb_times_R",
                "score_rule": "total_layer_map[l] = rgb_layer_map[l] * R_layer_map[l], then fuse",
                "ref": str(sample_dir / "ref.png"),
                "query": str(sample_dir / "query.png"),
                "Rref": str(sample_dir / "Rref.png"),
                "Rquery": str(sample_dir / "Rquery.png"),
                "rgb_heatmap": str(sample_dir / "rgb_heatmap.png"),
                "rgb_score_map": str(sample_dir / "rgb_score_map.png"),
                "R_heatmap": str(sample_dir / "R_heatmap.png"),
                "R_score_map": str(sample_dir / "R_score_map.png"),
                "heatmap": str(sample_dir / "heatmap.png"),
                "score_map": str(sample_dir / "score_map.png"),
                "pred_mask": str(sample_dir / "pred_mask.png"),
                "blend": str(sample_dir / "blend.png"),
            }
        )

    with open(output_dir / "summary.json", "w") as fd:
        json.dump(
            {
                "processed_samples": summary,
                "skipped_samples": skipped,
            },
            fd,
            indent=2,
        )


if __name__ == "__main__":
    main()
