import argparse
import json
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.datasets as datasets
import robust_defect_detection.models as models
from robust_defect_detection import evaluation, utils


def normalize_score_map(score_map):
    score_map = np.asarray(score_map, dtype=np.float32)
    score_map = score_map - score_map.min()
    score_map = score_map / max(score_map.max(), 1e-8)
    return score_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dataset-root", type=str, default="mydata")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    dataset_root = os.path.abspath(utils.resolve_path(args.dataset_root))

    device = args.device
    gpu_ids = args.gpu_ids
    if gpu_ids is None and str(device).startswith("cuda"):
        gpu_ids = [0]
    if str(device).startswith("cuda") and gpu_ids is not None and len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"

    model, checkpoint_data = models.load_checkpoint_model(checkpoint, device=device, gpu_ids=gpu_ids, verbose=False)
    inference_cfg = checkpoint_data["args"].get("inference", {})
    figsize = checkpoint_data["args"]["dataset"]["figsize"]
    loader = datasets.get_eval_loader(root=dataset_root, batch_size=1, num_workers=1, figsize=figsize)

    score_fn = model.module.compute_anomaly_maps if isinstance(model, torch.nn.DataParallel) else model.compute_anomaly_maps

    details = []
    for index, (ref_img, query_img, gt_mask) in enumerate(loader):
        ref_img = ref_img.to(device)
        query_img = query_img.to(device)
        with torch.no_grad():
            fused_map, _ = score_fn(
                ref_img,
                query_img,
                layer_weights=inference_cfg.get("layer-weights"),
                smooth_kernel=inference_cfg.get("gaussian-kernel", 5),
                smooth_sigma=inference_cfg.get("gaussian-sigma", 1.0),
            )
        score_map = fused_map.squeeze(0).detach().cpu().numpy()
        pred_mask = normalize_score_map(score_map) >= args.threshold
        gt_np = gt_mask.squeeze(0).numpy() > 0.5
        metrics = evaluation.change_mask_metric(pred_mask, gt_np)
        details.append(metrics)

    result = {
        key: float(np.mean([item[key] for item in details]))
        for key in ["precision", "recall", "accuracy", "f1_score", "iou"]
    }
    result["threshold"] = args.threshold
    print(json.dumps(result, indent=2))

    if args.output:
        output_path = utils.resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fd:
            json.dump(result, fd, indent=2)


if __name__ == "__main__":
    main()
