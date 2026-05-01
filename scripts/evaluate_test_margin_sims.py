"""Evaluate raw/trained per-layer margin similarities on a labelled test set.

The metrics mirror ``train_mydata.py``:

    pos_sim      = cos(ref_normal, query_normal) on foreground patches
    hard_neg_sim = cos(ref_normal, query_anomaly) on defect patches
    gap          = pos_sim - hard_neg_sim

Two labelled layouts are supported:

1. mydata-style object folders::

       obj_0001/ref.png
       obj_0001/ref_fg_mask.png                  # optional
       obj_0001/queries/*__normal__*.png
       obj_0001/queries/*__anomaly__*.png
       obj_0001/queries/*__anomaly__*_defect_mask.png

2. diagnose/testdata-style samples::

       Category/Sample/ref.png
       Category/Sample/query.png
       Category/Sample/gt.png

For layout 2, ``pos_sim`` is measured on clean foreground patches of the same
ref/query pair because no separate normal query exists.

Usage
-----
::

    python scripts/evaluate_test_margin_sims.py outputs/.../best.pth \
        --testdata-root testdata \
        --output outputs/.../test_margin_sims
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms.functional as tvff

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.models as models
from robust_defect_detection import utils


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_QUERY_RE = re.compile(
    r"^(?P<obj>obj_\d+)__"
    r"(?P<kind>normal|anomaly)__"
    r"(?P<defect>[^_]+(?:_[^_]+)*)__"
    r"light_(?P<light>\d+)\.png$"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--testdata-root", type=str, default="testdata")
    p.add_argument("--output", type=str, default="outputs/test_margin_sims")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--include-anomaly-clean-as-pos",
        action="store_true",
        help=(
            "Also count clean foreground patches from labelled anomaly pairs as "
            "pos_sim. By default pos_sim uses normal pairs only for mydata-style "
            "sets, matching train_mydata.py."
        ),
    )
    return p.parse_args()


def _load_rgb_tensor(path, figsize):
    img = Image.open(path).convert("RGB")
    img = img.resize((figsize[1], figsize[0]), Image.BILINEAR)
    t = tvff.to_tensor(img)
    return tvff.normalize(t, _IMAGENET_MEAN, _IMAGENET_STD).unsqueeze(0)


def _load_mask_tensor(path, figsize, default_value=None):
    if path is None or not Path(path).exists():
        if default_value is None:
            return None
        return torch.full(tuple(figsize), float(default_value), dtype=torch.float32)
    img = Image.open(path).convert("L")
    img = img.resize((figsize[1], figsize[0]), Image.NEAREST)
    return (tvff.to_tensor(img).squeeze(0) > 0.5).float()


def _pool_mask(mask, output_shape, device):
    mask = mask.to(device, non_blocking=True)
    return F.adaptive_avg_pool2d(mask.unsqueeze(0).unsqueeze(0), output_shape).squeeze(0).squeeze(0)


def _iter_mydata_style(root):
    root = Path(root)
    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("obj_")):
        ref = obj_dir / "ref.png"
        qdir = obj_dir / "queries"
        if not ref.exists() or not qdir.exists():
            continue
        fg = obj_dir / "ref_fg_mask.png"
        fg = fg if fg.exists() else None
        for q in sorted(qdir.iterdir()):
            if q.suffix.lower() != ".png" or q.stem.endswith("_defect_mask"):
                continue
            m = _QUERY_RE.match(q.name)
            if m is None:
                continue
            if m.group("kind") == "normal":
                yield {
                    "name": f"{obj_dir.name}/{q.name}",
                    "layout": "mydata",
                    "kind": "normal",
                    "ref": ref,
                    "query": q,
                    "fg": fg,
                    "defect": None,
                }
            else:
                defect = qdir / f"{q.stem}_defect_mask.png"
                if defect.exists():
                    yield {
                        "name": f"{obj_dir.name}/{q.name}",
                        "layout": "mydata",
                        "kind": "anomaly",
                        "ref": ref,
                        "query": q,
                        "fg": fg,
                        "defect": defect,
                    }


def _iter_diagnose_style(root):
    root = Path(root)
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sample_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            ref = sample_dir / "ref.png"
            query = sample_dir / "query.png"
            gt = sample_dir / "gt.png"
            if ref.exists() and query.exists() and gt.exists():
                yield {
                    "name": f"{cat_dir.name}/{sample_dir.name}",
                    "layout": "diagnose",
                    "kind": "labelled_pair",
                    "ref": ref,
                    "query": query,
                    "fg": None,
                    "defect": gt,
                }


def collect_samples(root):
    root = Path(root)
    samples = list(_iter_mydata_style(root))
    if samples:
        return samples
    return list(_iter_diagnose_style(root))


def _new_accumulator(layers):
    return {
        int(layer): {
            "pos_sum": 0.0,
            "pos_count": 0,
            "hard_neg_sum": 0.0,
            "hard_neg_count": 0,
        }
        for layer in layers
    }


def _add_mean(acc, key, values, mask):
    if not mask.any():
        return
    vals = values[mask]
    acc[f"{key}_sum"] += float(vals.sum().detach().cpu())
    acc[f"{key}_count"] += int(vals.numel())


def _finalize(acc):
    rows = []
    for layer, item in acc.items():
        pos = item["pos_sum"] / item["pos_count"] if item["pos_count"] else None
        hn = item["hard_neg_sum"] / item["hard_neg_count"] if item["hard_neg_count"] else None
        gap = (pos - hn) if pos is not None and hn is not None else None
        rows.append({
            "layer": layer,
            "pos_sim": pos,
            "hard_neg_sim": hn,
            "gap": gap,
            "n_pos_patches": item["pos_count"],
            "n_hard_neg_patches": item["hard_neg_count"],
        })
    return rows


@torch.no_grad()
def evaluate_encoder(
    encode_pair,
    samples,
    layers,
    figsize,
    cfg,
    device,
    include_anomaly_clean_as_pos=False,
):
    fg_thresh = float(cfg.get("foreground-thresh", 0.50))
    clean_thresh = float(cfg.get("patch-clean-thresh", 0.05))
    defect_thresh = float(cfg.get("patch-defect-thresh", 0.30))
    acc = _new_accumulator(layers)

    for sample in tqdm(samples, desc="samples", dynamic_ncols=True):
        ref_t = _load_rgb_tensor(sample["ref"], figsize).to(device)
        qry_t = _load_rgb_tensor(sample["query"], figsize).to(device)
        fg = _load_mask_tensor(sample["fg"], figsize, default_value=1.0)
        defect = _load_mask_tensor(sample["defect"], figsize, default_value=0.0)

        ref_feats, qry_feats = encode_pair(ref_t, qry_t)
        for layer, rf, qf in zip(layers, ref_feats, qry_feats):
            h, w = rf.shape[-2:]
            fg_p = _pool_mask(fg, (h, w), device) > fg_thresh
            def_p = _pool_mask(defect, (h, w), device)
            clean_p = (def_p < clean_thresh) & fg_p
            defect_p = (def_p > defect_thresh) & fg_p
            cos = F.cosine_similarity(rf, qf, dim=1).squeeze(0)
            layer_acc = acc[int(layer)]

            if sample["kind"] == "normal":
                _add_mean(layer_acc, "pos", cos, clean_p)
            elif sample["layout"] == "diagnose" or include_anomaly_clean_as_pos:
                _add_mean(layer_acc, "pos", cos, clean_p)

            if sample["kind"] != "normal":
                _add_mean(layer_acc, "hard_neg", cos, defect_p)

    return _finalize(acc)


def _write_outputs(out_dir, payload):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test_margin_sims_by_layer.json"
    csv_path = out_dir / "test_margin_sims_by_layer.csv"
    with open(json_path, "w") as fd:
        json.dump(payload, fd, indent=2)

    with open(csv_path, "w", newline="") as fd:
        writer = csv.DictWriter(
            fd,
            fieldnames=[
                "method",
                "layer",
                "pos_sim",
                "hard_neg_sim",
                "gap",
                "n_pos_patches",
                "n_hard_neg_patches",
            ],
        )
        writer.writeheader()
        for method in ("raw_dino", "trained"):
            for row in payload["metrics"][method]:
                writer.writerow({"method": method, **row})
    return json_path, csv_path


def _fmt(v):
    return "nan" if v is None else f"{v:.6f}"


def _find_args_json(checkpoint_path):
    ckpt_path = Path(utils.resolve_path(checkpoint_path))
    search_dirs = [ckpt_path.parent, *ckpt_path.parent.parents]
    project_root = Path(_PROJECT_ROOT).resolve()
    for directory in search_dirs:
        candidate = directory / "args.json"
        if candidate.exists():
            return candidate
        if directory.resolve() == project_root:
            break
    raise SystemExit(
        f"Could not find args.json from checkpoint path: {ckpt_path}. "
        "Expected args.json in the checkpoint directory or one of its parent directories."
    )


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    print("[eval_test_margin] loading checkpoint")
    trained, ckpt = models.load_checkpoint_model(args.checkpoint, device=device, verbose=False)
    trained.eval()

    args_json_path = _find_args_json(args.checkpoint)
    with open(args_json_path, "r") as fd:
        eval_args = json.load(fd)

    ckpt_layers = [int(layer) for layer in ckpt["args"]["model"]["layers"]]
    json_layers = [int(layer) for layer in eval_args["model"]["layers"]]
    if ckpt_layers != json_layers:
        raise SystemExit(
            f"args.json layers {json_layers} do not match checkpoint layers {ckpt_layers}."
        )

    model_args = ckpt["args"]["model"]
    if model_args.get("name") != "dino2 + contrast_learning":
        raise SystemExit(
            "evaluate_test_margin_sims.py only supports "
            "'dino2 + contrast_learning' checkpoints."
        )
    layers = [int(layer) for layer in eval_args["model"]["layers"]]
    figsize = tuple(int(v) for v in eval_args["dataset"]["figsize"])
    margin_cfg = eval_args.get("margin-loss", {})

    print("[eval_test_margin] building raw DINOv3 baseline")
    baseline = models.get_model(**model_args)
    baseline = models.wrap_model_for_gpus(baseline, device=device)
    baseline.eval()

    samples = collect_samples(utils.resolve_path(args.testdata_root))
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise SystemExit(
            "No labelled samples found. This script needs mydata-style "
            "normal/anomaly queries with defect masks, or diagnose-style "
            "ref/query/gt folders. Unlabelled mytestdata cannot produce a "
            "true hard_neg_sim."
        )

    layout = samples[0]["layout"]
    n_normal = sum(s["kind"] == "normal" for s in samples)
    n_anomaly = sum(s["kind"] != "normal" for s in samples)
    print(f"[eval_test_margin] layers : {layers}")
    print(f"[eval_test_margin] layout : {layout}")
    print(f"[eval_test_margin] samples: {len(samples)} ({n_normal} normal, {n_anomaly} labelled anomaly)")

    if isinstance(trained, torch.nn.DataParallel):
        trained_encode = trained.module.encode_pair
        raw_encode = baseline.module.backbone
    else:
        trained_encode = trained.encode_pair
        raw_encode = baseline.backbone

    print("[eval_test_margin] evaluating raw DINOv3")
    raw_rows = evaluate_encoder(
        raw_encode,
        samples,
        layers,
        figsize,
        margin_cfg,
        device,
        include_anomaly_clean_as_pos=args.include_anomaly_clean_as_pos,
    )
    print("[eval_test_margin] evaluating trained contrastive model")
    trained_rows = evaluate_encoder(
        trained_encode,
        samples,
        layers,
        figsize,
        margin_cfg,
        device,
        include_anomaly_clean_as_pos=args.include_anomaly_clean_as_pos,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "args_json": str(args_json_path),
        "testdata_root": str(args.testdata_root),
        "layers": layers,
        "layout": layout,
        "num_samples": len(samples),
        "num_normal_samples": n_normal,
        "num_labelled_anomaly_samples": n_anomaly,
        "pos_definition": (
            "normal-pair foreground patches"
            if layout == "mydata" and not args.include_anomaly_clean_as_pos
            else "clean foreground patches"
        ),
        "hard_neg_definition": "labelled defect foreground patches",
        "metrics": {
            "raw_dino": raw_rows,
            "trained": trained_rows,
        },
    }

    out_dir = Path(utils.resolve_path(args.output))
    json_path, csv_path = _write_outputs(out_dir, payload)

    print("\nmethod      layer  pos_sim   hard_neg_sim  gap")
    for method, rows in (("raw_dino", raw_rows), ("trained", trained_rows)):
        for row in rows:
            print(
                f"{method:<10} {row['layer']:>5}  "
                f"{_fmt(row['pos_sim']):>8}  "
                f"{_fmt(row['hard_neg_sim']):>12}  "
                f"{_fmt(row['gap']):>8}"
            )
    print(f"\n[eval_test_margin] wrote {json_path}")
    print(f"[eval_test_margin] wrote {csv_path}")


if __name__ == "__main__":
    main()
