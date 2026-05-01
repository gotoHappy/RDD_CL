"""Train a bidirectional cross-attention anomaly detector on top of a frozen
feature extractor.

Two modes:

1. **Contrastive backbone**  (``--backbone-ckpt path/to/best.pth`` or
   ``model.backbone-ckpt`` in YAML)
   Loads a previously-trained contrastive checkpoint (DINOv3 + LoRA +
   projector), freezes it, and trains the cross-attention head on top.

2. **Raw DINOv3 backbone**  (no checkpoint provided)
   Uses raw DINOv3 ViT-B/16 with an identity projector — equivalent to
   running cross-attention directly on the official pretrained features.
   No LoRA / no projector pre-training. Useful as an ablation showing how
   much the contrastive pretraining contributes.

Pipeline
--------
1. Build/load + freeze the feature extractor.
2. Wrap with :class:`CrossAttentionAnomalyDetector`:
   - stack of bidirectional cross-attention blocks
   - 1×1 conv decoder → single-channel logits → bilinear upsample to full
     resolution
3. Sample (ref, query, gt_mask) triplets from mydata: ref is a random normal
   query, query is a random anomaly query, gt_mask is the paired defect mask.
4. Train with BCE-with-logits against the binary GT mask. Output is a
   continuous anomaly score map (no binarisation needed).

Checkpoint format
-----------------
::

    {
      "args":            <full yaml config, with model.backbone-args resolved>,
      "extractor_state": <trained projector + LoRA weights from the contrastive
                          checkpoint, or {} when raw DINOv3 was used>,
      "model":           <state_dict of cross-attention + decoder only>,
      "optimizer":       ...,
      "lr_scheduler":    ...,
      "epoch":           ...,
      "logs":            ...,
    }

Run
---
::

    # On a contrastive backbone:
    python scripts/train_TwoCrossAttention.py \\
        scripts/configs/train_TwoCrossAttention.yml \\
        --backbone-ckpt outputs/.../best.pth

    # On raw DINOv3 (no checkpoint):
    python scripts/train_TwoCrossAttention.py \\
        scripts/configs/train_TwoCrossAttention.yml
"""

import argparse
import json
import math
import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from tqdm.auto import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.models as models
from robust_defect_detection import utils, utils_torch
from robust_defect_detection.datasets.mydata_change_detect import (
    build_mydata_change_detect_loader,
)
from robust_defect_detection.models.cross_attention import MultiLayerCrossAttentionAnomalyDetector


_dry = False
_seed = 123
_device = "cuda"
_verbose = False
_gpu_ids = None


# Fallback feature-extractor configuration used when no backbone checkpoint
# is provided. Reproduces a *raw DINOv3* setup: no LoRA, identity projector,
# DINO base weights loaded from the canonical local path. The user can
# override any of these by setting ``model.backbone-args`` in their YAML.
_DEFAULT_RAW_DINOV3_ARGS = {
    "name": "dino2 + contrast_learning",
    "dino-model": "dinov3_vitb16",
    "dino-weights-path": "/data2/baizeyu/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "layers": [9, 10, 11],
    "projector-type": "identity",
    "proj-out-dim": 256,                  # ignored when projector-type=identity
    "target-shp-row": 512,
    "target-shp-col": 512,
    "freeze-dino": True,
    "unfreeze-dino-last-n-layer": 0,
}


def xprint(*args):
    if _verbose:
        print(*args)


def resolve_project_path(path_like):
    return str(utils.resolve_path(path_like))


def _normalise_layer_list(layers, field_name="feature-layers"):
    if layers is None:
        return None
    if isinstance(layers, int):
        layers = [layers]
    out = [int(layer) for layer in layers]
    if not out:
        raise SystemExit(f"model.{field_name} must contain at least one layer")
    if len(set(out)) != len(out):
        raise SystemExit(f"model.{field_name} contains duplicates: {out}")
    return out


def _legacy_idx_to_layer(feature_layer_idx, backbone_layers):
    backbone_layers = [int(layer) for layer in backbone_layers]
    idx = int(feature_layer_idx)
    try:
        return backbone_layers[idx]
    except IndexError as exc:
        raise SystemExit(
            f"model.feature-layer-idx={idx} is out of range for backbone layers "
            f"{backbone_layers}. Prefer model.feature-layers with real layer IDs."
        ) from exc


def _resolve_feature_layers(model_cfg, backbone_layers, *, ckpt_restricted):
    backbone_layers = [int(layer) for layer in backbone_layers]
    feature_layers = _normalise_layer_list(model_cfg.get("feature-layers"))
    if feature_layers is None:
        feature_layers = [
            _legacy_idx_to_layer(model_cfg.get("feature-layer-idx", -1), backbone_layers)
        ]

    if ckpt_restricted:
        invalid = [layer for layer in feature_layers if layer not in backbone_layers]
        if invalid:
            raise SystemExit(
                f"model.feature-layers contains invalid layers {invalid}. "
                f"backbone-ckpt was trained with layers {backbone_layers}; "
                "feature-layers must be a subset of those real layer IDs."
            )
    return feature_layers


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_bce_loss(model, batch, pos_weight: float | None = None):
    """Per-pixel binary cross-entropy with logits.

    A configurable ``pos_weight`` up-weights the positive (defect) pixels
    because they're a small fraction of the image; without it the model
    can drive loss down by predicting all-zeros.

    The reported metrics split predictions across two key strata:
      • ``pos_pred_mean`` — mean prob at GT-positive pixels (defect pixels).
        Should rise toward 1 as training progresses.
      • ``neg_pred_mean_anom`` — mean prob at GT-negative pixels of
        ANOMALOUS-pair samples (background of defect-bearing images).
      • ``neg_pred_mean_norm`` — mean prob over ALL pixels of NORMAL-pair
        samples (where gt is fully zero). This is the *false-positive*
        signal — should stay low if the model isn't biased toward
        spurious activation on truly-normal pairs.
    """
    ref = batch["ref"].to(_device, non_blocking=True)
    qry = batch["query"].to(_device, non_blocking=True)
    gt = batch["gt_mask"].to(_device, non_blocking=True)         # (B, H, W) {0, 1}
    is_anom = batch["is_anomalous"].to(_device, non_blocking=True)  # (B,) {0, 1}

    logits = model(ref, qry)                                     # (B, H, W) raw

    if pos_weight is not None:
        pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pw)
    else:
        loss = F.binary_cross_entropy_with_logits(logits, gt)

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pos = gt > 0.5
        neg = ~pos
        anom_sample = is_anom.bool().view(-1, 1, 1).expand_as(gt)
        norm_sample = ~anom_sample
        # GT-negative pixels in anomalous-pair samples
        neg_in_anom = neg & anom_sample
        # All pixels in normal-pair samples (gt is fully zero, so neg ≡ all)
        all_in_norm = norm_sample
        report = {
            "loss": float(loss.item()),
            "pos_pred_mean": float(probs[pos].mean().item()) if pos.any() else 0.0,
            "neg_pred_mean_anom": float(probs[neg_in_anom].mean().item()) if neg_in_anom.any() else 0.0,
            "neg_pred_mean_norm": float(probs[all_in_norm].mean().item()) if all_in_norm.any() else 0.0,
            "n_pos_pixels": int(pos.sum().item()),
            "n_anom_pairs": int(is_anom.sum().item()),
            "n_norm_pairs": int((1 - is_anom).sum().item()),
        }
    return loss, report


# ---------------------------------------------------------------------------
# LR scheduler — same as train_mydata.py for consistency
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, final_scale=0.01):
        self.optimizer = optimizer
        self.warmup = max(int(warmup_epochs), 0)
        self.total = max(int(total_epochs), 1)
        self.final_scale = float(final_scale)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = -1

    def _scale(self, epoch):
        if self.warmup > 0 and epoch < self.warmup:
            return float(epoch + 1) / float(self.warmup)
        progress = (epoch - self.warmup + 1) / max(self.total - self.warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.final_scale + (1.0 - self.final_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch):
        self.last_epoch = int(epoch)
        s = self._scale(self.last_epoch)
        for base_lr, g in zip(self.base_lrs, self.optimizer.param_groups):
            g["lr"] = base_lr * s
        return s

    def state_dict(self):
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs,
                "warmup": self.warmup, "total": self.total, "final_scale": self.final_scale}

    def load_state_dict(self, s):
        self.last_epoch = s["last_epoch"]; self.base_lrs = s["base_lrs"]


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, scaler, loader, cfg, epoch_idx, total_epochs):
    model.train()
    losses, reports = [], []
    pos_weight = cfg.get("pos-weight")  # may be None

    bar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch_idx + 1}/{total_epochs}",
               position=1, leave=False, dynamic_ncols=True, ascii=False)

    for batch in bar:
        optimizer.zero_grad(set_to_none=True)
        loss, report = compute_bce_loss(model, batch, pos_weight=pos_weight)

        if not torch.isfinite(loss):
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        reports.append(report)

        bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss.item():.4f}",
            avg=f"{sum(losses)/max(len(losses),1):.4f}",
            pos=f"{report['pos_pred_mean']:.3f}",
            neg_a=f"{report['neg_pred_mean_anom']:.3f}",
            neg_n=f"{report['neg_pred_mean_norm']:.3f}",
        )

        if _dry:
            break

    bar.close()

    keys = ("pos_pred_mean", "neg_pred_mean_anom", "neg_pred_mean_norm")
    mean_report = {k: sum(r[k] for r in reports) / max(len(reports), 1) for k in keys}
    mean_report["n_pos_pixels"] = sum(r["n_pos_pixels"] for r in reports) / max(len(reports), 1)
    mean_report["n_anom_pairs"] = sum(r["n_anom_pairs"] for r in reports) / max(len(reports), 1)
    mean_report["n_norm_pairs"] = sum(r["n_norm_pairs"] for r in reports) / max(len(reports), 1)

    train_time = bar.format_dict.get("elapsed", 0.0)
    return sum(losses) / max(len(losses), 1), mean_report, train_time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_output_path_by_utc():
    return os.path.join(_PROJECT_ROOT, "outputs", utils.get_utc_time())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str)
    p.add_argument("--backbone-ckpt", type=str, default=None,
                   help="Path to a trained contrastive checkpoint (overrides "
                        "model.backbone-ckpt in the YAML). Required if not in YAML.")
    parsed = p.parse_args()

    with open(os.path.abspath(parsed.config), "r") as fd:
        args = yaml.safe_load(fd)

    args["dataset"]["root"] = resolve_project_path(args["dataset"]["root"])

    # backbone-ckpt resolution: CLI overrides YAML; missing is allowed and
    # falls back to a raw-DINOv3 backbone.
    backbone_ckpt = parsed.backbone_ckpt or args["model"].get("backbone-ckpt")
    args["model"]["backbone-ckpt"] = (
        resolve_project_path(backbone_ckpt) if backbone_ckpt else None
    )

    out = args["wandb"].get("output-path")
    args["wandb"]["output-path"] = resolve_project_path(out) if out else get_output_path_by_utc()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    utils_torch.seed_everything(_seed, verbose=_verbose)

    # ---- Build / load the feature extractor (will be frozen by the CA head) ----
    backbone_ckpt_path = args["model"].get("backbone-ckpt")
    if backbone_ckpt_path:
        # Mode 1: load trained contrastive checkpoint
        print(f"[train_CA] loading frozen backbone from checkpoint: {backbone_ckpt_path}")
        extractor, ext_ckpt = models.load_checkpoint_model(
            backbone_ckpt_path, device="cpu", gpu_ids=None, verbose=False,
        )
        if isinstance(extractor, nn.DataParallel):
            extractor = extractor.module
        backbone_args = ext_ckpt["args"]["model"]
        # Sidecar: trained projector + LoRA weights, stored alongside the CA
        # checkpoint so inference rebuilds without the original ckpt file.
        extractor_state = ext_ckpt["model"]
        feature_layers = _resolve_feature_layers(
            args["model"], backbone_args["layers"], ckpt_restricted=True,
        )
    else:
        # Mode 2: raw DINOv3 (no LoRA, identity projector). Honour
        # user-provided model.backbone-args in YAML, else use the canonical
        # default.
        backbone_args = args["model"].get("backbone-args") or dict(_DEFAULT_RAW_DINOV3_ARGS)
        default_layers = backbone_args.get("layers", _DEFAULT_RAW_DINOV3_ARGS["layers"])
        feature_layers = _resolve_feature_layers(
            args["model"], default_layers, ckpt_restricted=False,
        )
        # In raw-DINO mode there is no pretrained projector tied to a fixed
        # training-layer set, so rebuild the extractor to emit exactly the
        # real DINO layers requested by model.feature-layers.
        backbone_args = dict(backbone_args)
        backbone_args["layers"] = list(feature_layers)
        print("[train_CA] no backbone checkpoint provided — falling back to raw DINOv3")
        print(f"            dino-model={backbone_args.get('dino-model')}  "
              f"layers={backbone_args.get('layers')}  "
              f"projector={backbone_args.get('projector-type')}")
        extractor = models.get_model(**backbone_args)
        # No trained weights to carry over — extractor relies entirely on
        # DINOv3 base weights (loaded from disk inside get_model).
        extractor_state = {}

    args["model"]["backbone-args"] = backbone_args
    args["model"]["feature-layers"] = list(feature_layers)
    print(f"[train_CA] cross-attention feature layers: {feature_layers}")

    # ---- Build the cross-attention model wrapping the loaded extractor ----
    target_shp = (
        int(args["model"]["target-shp-row"]),
        int(args["model"]["target-shp-col"]),
    )
    model = MultiLayerCrossAttentionAnomalyDetector(
        feature_extractor=extractor,
        feature_layers=feature_layers,
        embed_dim=args["model"].get("embed-dim"),
        num_heads=int(args["model"].get("num-heads", 4)),
        num_blocks=int(args["model"].get("num-blocks", 2)),
        ffn_ratio=float(args["model"].get("ffn-ratio", 4.0)),
        dropout=float(args["model"].get("dropout", 0.0)),
        target_shp=target_shp,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_CA] cross-attention + decoder trainable params: {n_train:,}")
    model = models.wrap_model_for_gpus(model, device=_device, gpu_ids=_gpu_ids)

    # ---- Data ----
    ds_cfg = args["dataset"]
    loader = build_mydata_change_detect_loader(
        mydata_root=ds_cfg["root"],
        figsize=tuple(ds_cfg["figsize"]),
        batch_size=int(ds_cfg["batch-size"]),
        num_workers=int(ds_cfg.get("num-workers", 2)),
        spatial_scale=tuple(ds_cfg.get("spatial-scale", [0.85, 1.0])),
        normal_pair_ratio=float(ds_cfg.get("normal-pair-ratio", 0.3)),
    )
    xprint(f"[train_CA] mydata change-detect loader: {len(loader)} steps/epoch  "
           f"(normal-pair-ratio={ds_cfg.get('normal-pair-ratio', 0.3)})")

    # ---- Optimizer / scheduler ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters — is feature_extractor wrongly unfrozen?")

    opt_cfg = args["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(opt_cfg["learn-rate"]),
        weight_decay=float(opt_cfg.get("weight-decay", 0.05)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=int(opt_cfg.get("warmup-epoch", 0)),
        total_epochs=int(opt_cfg["epochs"]),
        final_scale=float(opt_cfg.get("final-scale", 0.01)),
    )
    use_scaler = bool(opt_cfg.get("grad-scaler", False)) and str(_device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    # ---- Wandb ----
    wandb_mode = "disabled" if args["wandb"].get("mode", "online") == "disabled" else "online"
    wandb.init(project=args["wandb"]["project"], name=args["wandb"]["name"], config=args, mode=wandb_mode)

    best_loss = float("inf")
    checkpoint = None
    total_epochs = int(opt_cfg["epochs"])

    epoch_bar = tqdm(range(total_epochs), total=total_epochs, desc="Epochs",
                     position=0, leave=True, dynamic_ncols=True, ascii=False)

    for epoch in epoch_bar:
        scheduler.step(epoch)
        loss, report, train_time = train_one_epoch(
            model, optimizer, scaler, loader, args["loss"], epoch, total_epochs,
        )

        logs = {
            "loss": loss,
            "epoch": epoch,
            "time.train": train_time,
            "ca": report,
        }

        # Save only the cross-attention + decoder trainable state. The
        # frozen extractor's trained weights live in extractor_state (saved
        # alongside) so inference can fully reconstruct without external
        # files.
        checkpoint = {
            "model": utils_torch.get_grad_required_state(model),
            "extractor_state": extractor_state,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "logs": logs,
        }

        with open(os.path.join(args["wandb"]["output-path"], "logs", f"{epoch}.layer.pkl"), "wb") as fd:
            pickle.dump(logs, fd)

        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "ca/pos_pred_mean": report["pos_pred_mean"],
            "ca/neg_pred_mean_anom": report["neg_pred_mean_anom"],
            "ca/neg_pred_mean_norm": report["neg_pred_mean_norm"],
            "n_pos_pixels": report["n_pos_pixels"],
            "n_anom_pairs": report["n_anom_pairs"],
            "n_norm_pairs": report["n_norm_pairs"],
            "time/train": train_time,
            "lr": optimizer.param_groups[0]["lr"],
        })

        epoch_bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss:.4f}",
            pos=f"{report['pos_pred_mean']:.3f}",
            neg_a=f"{report['neg_pred_mean_anom']:.3f}",
            neg_n=f"{report['neg_pred_mean_norm']:.3f}",
            best=f"{best_loss:.4f}" if best_loss < float("inf") else "n/a",
        )

        save_freq = int(args["wandb"].get("save-checkpoint-freq", 0))
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(checkpoint, os.path.join(args["wandb"]["output-path"], "checkpoints", f"{epoch}.layer.pth"))
        if loss < best_loss:
            best_loss = loss
            torch.save(checkpoint, os.path.join(args["wandb"]["output-path"], "best.pth"))

        if _dry:
            break

    epoch_bar.close()

    if checkpoint is not None:
        torch.save(checkpoint, os.path.join(args["wandb"]["output-path"], "last.pth"))

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    env = args["environment"]
    _dry = bool(env["dry"])
    _seed = int(env["seed"])
    _verbose = bool(env["verbose"])
    _device = env["device"]
    _gpu_ids = env.get("gpu-ids")
    if _gpu_ids is None:
        _gpu_ids = [0] if str(_device).startswith("cuda") else []
    if str(_device).startswith("cuda") and len(_gpu_ids) == 1:
        _device = f"cuda:{_gpu_ids[0]}"

    output_path = os.path.abspath(args["wandb"]["output-path"])
    args["wandb"]["output-path"] = output_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
    with open(os.path.join(output_path, "args.json"), "w") as fd:
        json.dump(args, fd, indent=4)

    main(args)
