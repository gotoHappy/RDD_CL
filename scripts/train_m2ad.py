"""
Training script for M2AD contrastive learning with InfoNCE loss.

Pipeline:
    1. Each batch: B objects × 3 images (N1, N2, A1) = 3B images.
    2. DINOv3 frozen + per-layer projector → L2-normalised patch features.
    3. Per layer, build anchor / positive / hard-neg / in-batch-neg sets and
       compute InfoNCE. Average across layers.

See Training_Program.md §4 for the loss derivation.
"""

import argparse
import json
import math
import os
import pickle
import sys

import torch
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
from robust_defect_detection.datasets.m2ad import build_m2ad_triplet_loader


# ---------------------------------------------------------------------------
# Global runtime config (populated from YAML environment block)
# ---------------------------------------------------------------------------

_dry = False
_seed = 123
_device = "cuda"
_verbose = False
_gpu_ids = None


def xprint(*args):
    if _verbose:
        print(*args)


def resolve_project_path(path_like):
    return str(utils.resolve_path(path_like))


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def _pool_mask(mask, output_shape):
    """Avg-pool a (B, H, W) mask down to (B, h, w)."""
    return F.adaptive_avg_pool2d(mask.unsqueeze(1).float(), output_shape).squeeze(1)


def _compute_layer_infonce(feat_n1, feat_n2, feat_a1, fg_binary, def_labels, owner, cfg):
    """
    One-layer InfoNCE loss.

    Inputs:
        feat_n1 / feat_n2 / feat_a1 : (B, C, h, w) L2-normalised patch features.
        fg_binary   : (B, h, w) bool  — foreground patches.
        def_labels  : (B, h, w) int64 — 0 clean / 1 defect / -1 ignore.
        owner       : (B,)      int64 — object identity per batch item.
    """
    B, C, h, w = feat_n1.shape
    tau = float(cfg["temperature"])
    device = feat_n1.device

    anchor_list, pos_list, hard_list = [], [], []
    h_mask_list, owner_list, valid_list = [], [], []

    for b in range(B):
        ij = fg_binary[b].nonzero(as_tuple=False)  # (K, 2)
        K = ij.shape[0]
        if K == 0:
            continue
        ri, ci = ij[:, 0], ij[:, 1]

        anchor_list.append(feat_n1[b, :, ri, ci].T)   # (K, C)
        pos_list.append(feat_n2[b, :, ri, ci].T)      # (K, C)
        hard_list.append(feat_a1[b, :, ri, ci].T)     # (K, C)
        h_mask_list.append(def_labels[b, ri, ci] == 1)   # (K,)
        valid_list.append(def_labels[b, ri, ci] != -1)   # (K,)
        owner_list.append(
            torch.full((K,), int(owner[b].item()), dtype=torch.long, device=device)
        )

    if not anchor_list:
        zero = feat_n1.new_tensor(0.0)
        return zero, _zero_metrics()

    Za = torch.cat(anchor_list).float()   # (M, C)
    Zp = torch.cat(pos_list).float()
    Zh = torch.cat(hard_list).float()
    H_mask = torch.cat(h_mask_list)       # (M,) bool
    valid = torch.cat(valid_list)         # (M,) bool
    owner_t = torch.cat(owner_list)       # (M,)

    # Positive similarity (scaled by 1/τ)
    S_pos = (Za * Zp).sum(-1) / tau       # (M,)

    # Hard-negative similarity — set to -inf where there is no hard neg.
    S_hard_raw = (Za * Zh).sum(-1) / tau  # (M,)
    neg_inf = S_hard_raw.new_tensor(float("-inf"))
    S_hard = torch.where(H_mask, S_hard_raw, neg_inf)

    # In-batch negative similarity matrix.
    S_all = (Za @ Za.T) / tau             # (M, M)
    same_obj = owner_t.unsqueeze(0) == owner_t.unsqueeze(1)
    S_all = S_all.masked_fill(same_obj, float("-inf"))

    # log-denominator via logsumexp over [pos, hard, in-batch]
    denom_parts = torch.cat(
        [S_pos.unsqueeze(1), S_hard.unsqueeze(1), S_all], dim=1
    )  # (M, M+2)
    log_denom = torch.logsumexp(denom_parts, dim=1)  # (M,)

    losses = -S_pos + log_denom  # (M,)

    if valid.sum() == 0:
        zero = feat_n1.new_tensor(0.0)
        return zero, _zero_metrics()

    loss = losses[valid].mean()

    # Monitoring metrics (cosine-space, not scaled by τ).
    with torch.no_grad():
        pos_cos = (Za * Zp).sum(-1)
        hard_cos = (Za * Zh).sum(-1)
        inbatch_cos = S_all * tau  # undo τ
        finite = torch.isfinite(inbatch_cos)
        metrics = {
            "pos_sim": float(pos_cos[valid].mean().item()),
            "hard_neg_sim": float(hard_cos[H_mask].mean().item()) if H_mask.any() else 0.0,
            "inbatch_neg_sim": float(inbatch_cos[finite].mean().item()) if finite.any() else 0.0,
            "n_valid": int(valid.sum().item()),
            "n_defect": int(H_mask.sum().item()),
        }
    return loss, metrics


def _zero_metrics():
    return {
        "pos_sim": 0.0,
        "hard_neg_sim": 0.0,
        "inbatch_neg_sim": 0.0,
        "n_valid": 0,
        "n_defect": 0,
    }


def compute_infonce_loss(model, batch, cfg):
    """Per-layer InfoNCE, weighted & averaged."""
    n1 = batch["n1"].to(_device, non_blocking=True)
    n2 = batch["n2"].to(_device, non_blocking=True)
    a1 = batch["a1"].to(_device, non_blocking=True)
    fg_mask = batch["fg_mask"].to(_device, non_blocking=True)          # (B, H, W)
    defect_mask = batch["defect_mask"].to(_device, non_blocking=True)  # (B, H, W)
    owner = batch["object_idx"].to(_device, non_blocking=True)         # (B,)

    encode = (
        model.module.encode_single
        if isinstance(model, torch.nn.DataParallel)
        else model.encode_single
    )
    z_n1 = encode(n1)
    z_n2 = encode(n2)
    z_a1 = encode(a1)

    num_layers = len(z_n1)
    layer_weights = cfg.get("layer-loss-weights") or [1.0] * num_layers
    wt = torch.tensor(layer_weights[:num_layers], dtype=torch.float32, device=_device)
    wt = wt / wt.sum().clamp_min(1e-8)

    fg_thresh = float(cfg["foreground-thresh"])
    clean_thresh = float(cfg["patch-clean-thresh"])
    defect_thresh = float(cfg["patch-defect-thresh"])

    total_loss = torch.tensor(0.0, device=_device)
    agg = {"pos_sim": 0.0, "hard_neg_sim": 0.0, "inbatch_neg_sim": 0.0, "n_valid": 0, "n_defect": 0}

    for li, (fn1, fn2, fa1) in enumerate(zip(z_n1, z_n2, z_a1)):
        h, w_feat = fn1.shape[-2:]
        fg_p = _pool_mask(fg_mask, (h, w_feat)) > fg_thresh         # (B, h, w) bool
        def_p = _pool_mask(defect_mask, (h, w_feat))                # (B, h, w) float

        def_labels = torch.full(def_p.shape, -1, dtype=torch.long, device=_device)
        def_labels[(def_p < clean_thresh) & fg_p] = 0
        def_labels[(def_p > defect_thresh) & fg_p] = 1

        layer_loss, metrics = _compute_layer_infonce(
            fn1, fn2, fa1, fg_p, def_labels, owner, cfg
        )
        total_loss = total_loss + wt[li] * layer_loss

        # Weighted average of monitoring metrics
        w_py = float(wt[li].item())
        agg["pos_sim"] += w_py * metrics["pos_sim"]
        agg["hard_neg_sim"] += w_py * metrics["hard_neg_sim"]
        agg["inbatch_neg_sim"] += w_py * metrics["inbatch_neg_sim"]
        agg["n_valid"] = max(agg["n_valid"], metrics["n_valid"])
        agg["n_defect"] = max(agg["n_defect"], metrics["n_defect"])

    return total_loss, agg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, scaler, loader, cfg, epoch_idx, total_epochs):
    model.train()
    losses, reports = [], []

    bar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch_idx + 1}/{total_epochs}",
        position=1,
        leave=False,
        dynamic_ncols=True,
        ascii=False,
    )

    for batch in bar:
        optimizer.zero_grad()
        loss, report = compute_infonce_loss(model, batch, cfg)

        if not torch.isfinite(loss):
            # Skip pathological batch (no valid anchors, etc.)
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

        lr = optimizer.param_groups[0]["lr"]
        mean_loss = sum(losses) / max(len(losses), 1)
        bar.set_postfix(
            lr=f"{lr:.2e}",
            loss=f"{loss.item():.4f}",
            avg=f"{mean_loss:.4f}",
            pos=f"{report['pos_sim']:.3f}",
            hn=f"{report['hard_neg_sim']:.3f}",
            ib=f"{report['inbatch_neg_sim']:.3f}",
        )

        if _dry:
            break

    bar.close()

    mean_report = {
        k: sum(r[k] for r in reports) / max(len(reports), 1)
        for k in ["pos_sim", "hard_neg_sim", "inbatch_neg_sim"]
    }
    mean_report["n_valid"] = sum(r["n_valid"] for r in reports) / max(len(reports), 1)
    mean_report["n_defect"] = sum(r["n_defect"] for r in reports) / max(len(reports), 1)

    train_time = bar.format_dict.get("elapsed", 0.0)
    return sum(losses) / max(len(losses), 1), mean_report, train_time


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay (per-epoch)
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, final_scale=0.0):
        self.optimizer = optimizer
        self.warmup = max(int(warmup_epochs), 0)
        self.total = max(int(total_epochs), 1)
        self.final_scale = float(final_scale)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.epoch = 0

    def _scale(self):
        if self.epoch < self.warmup:
            return (self.epoch + 1) / max(self.warmup, 1)
        progress = (self.epoch - self.warmup) / max(self.total - self.warmup, 1)
        cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return self.final_scale + (1.0 - self.final_scale) * cos

    def step(self):
        self.epoch += 1
        s = self._scale()
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * s

    def state_dict(self):
        return {"epoch": self.epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.base_lrs = state["base_lrs"]


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def get_output_path_by_utc():
    return os.path.join(_PROJECT_ROOT, "outputs", utils.get_utc_time())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parsed = parser.parse_args()
    with open(os.path.abspath(parsed.config), "r") as fd:
        args = yaml.safe_load(fd)

    ds = args["dataset"]
    ds["m2ad-root"] = resolve_project_path(ds["m2ad-root"])
    ds["json-path"] = resolve_project_path(ds["json-path"])
    if ds.get("mask-root"):
        ds["mask-root"] = resolve_project_path(ds["mask-root"])
    if ds.get("dtd-root"):
        ds["dtd-root"] = resolve_project_path(ds["dtd-root"])

    out = args["wandb"].get("output-path")
    args["wandb"]["output-path"] = resolve_project_path(out) if out else get_output_path_by_utc()
    return args


def main(args):
    utils_torch.seed_everything(_seed, verbose=_verbose)

    # Model
    model = models.get_model(**args["model"])
    model = models.wrap_model_for_gpus(model, device=_device, gpu_ids=_gpu_ids)

    # Data
    ds_cfg = args["dataset"]
    loader = build_m2ad_triplet_loader(
        m2ad_root=ds_cfg["m2ad-root"],
        json_path=ds_cfg["json-path"],
        mask_root=ds_cfg.get("mask-root"),
        dtd_root=ds_cfg.get("dtd-root"),
        split=ds_cfg.get("split", "train"),
        figsize=ds_cfg["figsize"],
        batch_size=ds_cfg["batch-size"],
        num_workers=ds_cfg["num-workers"],
        min_lights_per_view=ds_cfg.get("min-lights-per-view", 2),
        perlin_octaves=tuple(ds_cfg.get("perlin-octaves", [4, 6])),
        perlin_scale=tuple(ds_cfg.get("perlin-scale", [2.0, 6.0])),
        min_area_ratio=ds_cfg.get("min-area-ratio", 0.05),
        max_area_ratio=ds_cfg.get("max-area-ratio", 0.30),
    )

    # Optimizer — AdamW on trainable params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = args["optimizer"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(opt_cfg["learn-rate"]),
        weight_decay=float(opt_cfg.get("weight-decay", 0.05)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=int(opt_cfg["warmup-epoch"]),
        total_epochs=int(opt_cfg["epochs"]),
        final_scale=float(opt_cfg.get("final-scale", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(opt_cfg.get("grad-scaler", False)))

    # Wandb
    wandb_mode = "disabled" if args["wandb"].get("mode", "online") == "disabled" else "online"
    wandb.init(
        project=args["wandb"]["project"],
        name=args["wandb"]["name"],
        config=args,
        mode=wandb_mode,
    )

    best_loss = float("inf")
    checkpoint = None

    epoch_bar = tqdm(
        range(int(opt_cfg["epochs"])),
        total=int(opt_cfg["epochs"]),
        desc="Epochs",
        position=0,
        leave=True,
        dynamic_ncols=True,
        ascii=False,
    )

    for epoch in epoch_bar:
        loss, report, train_time = train_one_epoch(
            model,
            optimizer,
            scaler,
            loader,
            args["contrastive-loss"],
            epoch,
            int(opt_cfg["epochs"]),
        )
        scheduler.step()

        logs = {
            "loss": loss,
            "epoch": epoch,
            "time.train": train_time,
            "contrastive": report,
        }
        checkpoint = {
            "model": utils_torch.get_grad_required_state(model),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "logs": logs,
        }

        with open(os.path.join(args["wandb"]["output-path"], "logs", f"{epoch}.layer.pkl"), "wb") as fd:
            pickle.dump(logs, fd)

        wandb.log(
            {
                "epoch": epoch,
                "loss": loss,
                "loss/pos_sim": report["pos_sim"],
                "loss/hard_neg_sim": report["hard_neg_sim"],
                "loss/inbatch_neg_sim": report["inbatch_neg_sim"],
                "n_valid": report["n_valid"],
                "n_defect": report["n_defect"],
                "time/train": train_time,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        epoch_bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss:.4f}",
            pos=f"{report['pos_sim']:.3f}",
            hn=f"{report['hard_neg_sim']:.3f}",
            ib=f"{report['inbatch_neg_sim']:.3f}",
            best=f"{best_loss:.4f}" if best_loss < float("inf") else "n/a",
        )

        save_freq = int(args["wandb"]["save-checkpoint-freq"])
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(
                checkpoint,
                os.path.join(args["wandb"]["output-path"], "checkpoints", f"{epoch}.layer.pth"),
            )
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
    _dry = env["dry"]
    _seed = env["seed"]
    _verbose = env["verbose"]
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
