"""
Train on mydata only with real defect masks.

This entry point is intentionally independent from the older
``train_contrastive.py`` path:

- data:        mydata only, via ``MyDataTripletDataset``
- backbone:    configurable, default DINOv3 ViT-B/16
- loss:        pair-margin loss over foreground patches, no M2AD and no online
               pseudo anomaly synthesis
- checkpoint:  same ``checkpoint['args']['model']`` format used by the existing
               inference / visualization scripts
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
from robust_defect_detection.datasets.mydata_triplet import build_mydata_triplet_loader


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
# Pair-margin loss
# ---------------------------------------------------------------------------

def _pool_mask(mask, output_shape):
    """Avg-pool a (B, H, W) mask down to (B, h, w)."""
    return F.adaptive_avg_pool2d(mask.unsqueeze(1).float(), output_shape).squeeze(1)


def _zero_metrics():
    return {
        "pos_sim": 0.0,
        "hard_neg_sim": 0.0,
        "gap": 0.0,
        "n_clean": 0,
        "n_defect": 0,
        "loss_clean": 0.0,
        "loss_defect": 0.0,
    }


def _compute_layer_margin_loss(feat_n1, feat_n2, feat_a1, fg_binary, def_labels, cfg):
    """
    One-layer pair-margin loss.

    Clean foreground patches are encouraged to keep N1 and N2 similar.
    Defect patches are encouraged to make N1 closer to N2 than to A1.
    """
    margin_triplet = float(cfg.get("margin-triplet", 0.3))
    margin_positive = float(cfg.get("margin-positive", 0.95))
    clean_weight = float(cfg.get("clean-weight", 1.0))
    defect_weight = float(cfg.get("defect-weight", 3.0))

    pos_sim_map = (feat_n1 * feat_n2).sum(dim=1)
    hard_sim_map = (feat_n1 * feat_a1).sum(dim=1)

    fg_valid = fg_binary & (def_labels != -1)
    clean_mask = fg_valid & (def_labels == 0)
    defect_mask = fg_valid & (def_labels == 1)

    if clean_mask.any():
        loss_clean = F.relu(margin_positive - pos_sim_map[clean_mask]).mean()
    else:
        loss_clean = feat_n1.new_tensor(0.0)

    if defect_mask.any():
        pos_def = pos_sim_map[defect_mask]
        hard_def = hard_sim_map[defect_mask]
        gap = pos_def - hard_def
        loss_defect = F.relu(margin_triplet - gap).mean()
    else:
        loss_defect = feat_n1.new_tensor(0.0)

    weight_sum = max(clean_weight + defect_weight, 1e-8)
    loss = (clean_weight * loss_clean + defect_weight * loss_defect) / weight_sum

    with torch.no_grad():
        metrics = {
            "pos_sim": float(pos_sim_map[fg_valid].mean().item()) if fg_valid.any() else 0.0,
            "hard_neg_sim": float(hard_sim_map[defect_mask].mean().item()) if defect_mask.any() else 0.0,
            "gap": float((pos_sim_map[defect_mask] - hard_sim_map[defect_mask]).mean().item())
            if defect_mask.any() else 0.0,
            "n_clean": int(clean_mask.sum().item()),
            "n_defect": int(defect_mask.sum().item()),
            "loss_clean": float(loss_clean.item()),
            "loss_defect": float(loss_defect.item()),
        }
    return loss, metrics


def compute_margin_loss(model, batch, cfg):
    n1 = batch["n1"].to(_device, non_blocking=True)
    n2 = batch["n2"].to(_device, non_blocking=True)
    a1 = batch["a1"].to(_device, non_blocking=True)
    fg_mask = batch["fg_mask"].to(_device, non_blocking=True)
    defect_mask = batch["defect_mask"].to(_device, non_blocking=True)

    encode = model.module.encode_single if isinstance(model, torch.nn.DataParallel) else model.encode_single
    z_n1 = encode(n1)
    z_n2 = encode(n2)
    z_a1 = encode(a1)

    num_layers = len(z_n1)
    layer_weights = cfg.get("layer-loss-weights") or [1.0] * num_layers
    if len(layer_weights) < num_layers:
        raise ValueError(
            f"layer-loss-weights has {len(layer_weights)} entries, but model produced {num_layers} layers"
        )
    wt = torch.tensor(layer_weights[:num_layers], dtype=torch.float32, device=_device)
    wt = wt / wt.sum().clamp_min(1e-8)

    fg_thresh = float(cfg["foreground-thresh"])
    clean_thresh = float(cfg["patch-clean-thresh"])
    defect_thresh = float(cfg["patch-defect-thresh"])

    total_loss = torch.tensor(0.0, device=_device)
    agg = _zero_metrics()

    for weight, fn1, fn2, fa1 in zip(wt, z_n1, z_n2, z_a1):
        h, w_feat = fn1.shape[-2:]
        fg_p = _pool_mask(fg_mask, (h, w_feat)) > fg_thresh
        def_p = _pool_mask(defect_mask, (h, w_feat))

        def_labels = torch.full(def_p.shape, -1, dtype=torch.long, device=_device)
        def_labels[(def_p < clean_thresh) & fg_p] = 0
        def_labels[(def_p > defect_thresh) & fg_p] = 1

        layer_loss, metrics = _compute_layer_margin_loss(fn1, fn2, fa1, fg_p, def_labels, cfg)
        total_loss = total_loss + weight * layer_loss

        w_py = float(weight.item())
        for key in ("pos_sim", "hard_neg_sim", "gap", "loss_clean", "loss_defect"):
            agg[key] += w_py * metrics[key]
        agg["n_clean"] = max(agg["n_clean"], metrics["n_clean"])
        agg["n_defect"] = max(agg["n_defect"], metrics["n_defect"])

    return total_loss, agg


# ---------------------------------------------------------------------------
# Training
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
        optimizer.zero_grad(set_to_none=True)
        loss, report = compute_margin_loss(model, batch, cfg)

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

        mean_loss = sum(losses) / max(len(losses), 1)
        bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss.item():.4f}",
            avg=f"{mean_loss:.4f}",
            pos=f"{report['pos_sim']:.3f}",
            hn=f"{report['hard_neg_sim']:.3f}",
            gap=f"{report['gap']:.3f}",
            nd=f"{report['n_defect']}",
        )

        if _dry:
            break

    bar.close()

    keys = ("pos_sim", "hard_neg_sim", "gap", "loss_clean", "loss_defect")
    mean_report = {key: sum(r[key] for r in reports) / max(len(reports), 1) for key in keys}
    mean_report["n_clean"] = sum(r["n_clean"] for r in reports) / max(len(reports), 1)
    mean_report["n_defect"] = sum(r["n_defect"] for r in reports) / max(len(reports), 1)

    train_time = bar.format_dict.get("elapsed", 0.0)
    return sum(losses) / max(len(losses), 1), mean_report, train_time


class WarmupCosineScheduler:
    """Epoch-level warmup + cosine decay, applied before each epoch."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, final_scale=0.01):
        self.optimizer = optimizer
        self.warmup = max(int(warmup_epochs), 0)
        self.total = max(int(total_epochs), 1)
        self.final_scale = float(final_scale)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    def _scale(self, epoch):
        if self.warmup > 0 and epoch < self.warmup:
            return float(epoch + 1) / float(self.warmup)
        progress = (epoch - self.warmup + 1) / max(self.total - self.warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_scale + (1.0 - self.final_scale) * cosine

    def step(self, epoch):
        self.last_epoch = int(epoch)
        scale = self._scale(self.last_epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale
        return scale

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "base_lrs": self.base_lrs,
            "warmup": self.warmup,
            "total": self.total,
            "final_scale": self.final_scale,
        }

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.base_lrs = state["base_lrs"]


def get_output_path_by_utc():
    return os.path.join(_PROJECT_ROOT, "outputs", utils.get_utc_time())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parsed = parser.parse_args()
    with open(os.path.abspath(parsed.config), "r") as fd:
        args = yaml.safe_load(fd)

    args["dataset"]["root"] = resolve_project_path(args["dataset"]["root"])
    out = args["wandb"].get("output-path")
    args["wandb"]["output-path"] = resolve_project_path(out) if out else get_output_path_by_utc()
    return args


def main(args):
    utils_torch.seed_everything(_seed, verbose=_verbose)

    model = models.get_model(**args["model"])
    model = models.wrap_model_for_gpus(model, device=_device, gpu_ids=_gpu_ids)

    ds_cfg = args["dataset"]
    loader = build_mydata_triplet_loader(
        mydata_root=ds_cfg["root"],
        figsize=tuple(ds_cfg["figsize"]),
        batch_size=int(ds_cfg["batch-size"]),
        num_workers=int(ds_cfg.get("num-workers", 2)),
        spatial_scale=tuple(ds_cfg.get("spatial-scale", [0.8, 1.0])),
    )
    xprint(f"mydata loader: {len(loader)} steps per epoch")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")

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

    wandb_mode = "disabled" if args["wandb"].get("mode", "online") == "disabled" else "online"
    wandb.init(
        project=args["wandb"]["project"],
        name=args["wandb"]["name"],
        config=args,
        mode=wandb_mode,
    )

    best_loss = float("inf")
    checkpoint = None
    total_epochs = int(opt_cfg["epochs"])

    epoch_bar = tqdm(
        range(total_epochs),
        total=total_epochs,
        desc="Epochs",
        position=0,
        leave=True,
        dynamic_ncols=True,
        ascii=False,
    )

    for epoch in epoch_bar:
        lr_scale = scheduler.step(epoch)
        loss, report, train_time = train_one_epoch(
            model,
            optimizer,
            scaler,
            loader,
            args["margin-loss"],
            epoch,
            total_epochs,
        )

        logs = {
            "loss": loss,
            "epoch": epoch,
            "time.train": train_time,
            "lr_scale": lr_scale,
            "margin": report,
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

        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "loss/clean": report["loss_clean"],
            "loss/defect": report["loss_defect"],
            "margin/pos_sim": report["pos_sim"],
            "margin/hard_neg_sim": report["hard_neg_sim"],
            "margin/gap": report["gap"],
            "n_clean": report["n_clean"],
            "n_defect": report["n_defect"],
            "time/train": train_time,
            "lr": optimizer.param_groups[0]["lr"],
            "lr_scale": lr_scale,
        })

        epoch_bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss:.4f}",
            pos=f"{report['pos_sim']:.3f}",
            hn=f"{report['hard_neg_sim']:.3f}",
            gap=f"{report['gap']:.3f}",
            best=f"{best_loss:.4f}" if best_loss < float("inf") else "n/a",
        )

        save_freq = int(args["wandb"].get("save-checkpoint-freq", 0))
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
