"""
Training script: M2AD as cross-illumination invariance source (N1-N2 pairs,
**no synthetic anomaly**) + mydata as real-defect triplet source.

Design rationale
----------------
Previous experiments showed mydata-only training beat both M2AD-only and
the mixed (M2AD-with-synth + mydata) setups. Synthetic Perlin/DTD anomalies
don't align with the real defect distribution and actively dilute the
signal. But M2AD's multi-light same-object imagery is still valuable —
it's strictly a lighting-invariance training source. This script uses it
purely for that, dropping the synthetic anomaly branch entirely.

Per step:
  - M2AD batch:    (N1, N2, fg_mask)                        → L_clean only
  - mydata batch:  (N1, N2, A1, fg_mask, defect_mask)       → L_clean + L_defect

Loss per layer (foreground patches only, ignore-labels excluded):
    L_clean  = ReLU(β − pos_sim)                            # per-source
    L_defect = ReLU(α − (pos_sim − hard_neg_sim))           # mydata defect anchors

Layer-weighted mean across the requested transformer blocks; source-weighted
combination in config. No in-batch cross-object negatives anywhere.
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
from robust_defect_detection.datasets.m2ad import build_m2ad_pair_loader
from robust_defect_detection.datasets.mydata_triplet import MyDataTripletDataset
from torch.utils.data import DataLoader


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
# Dual-source loader
# ---------------------------------------------------------------------------

class DualSourceLoader:
    """Yields ``{'m2ad': batch_m, 'mydata': batch_y}`` per step.
    The longer stream drives iteration count; the shorter one is cycled.
    """

    def __init__(self, m2ad_loader, mydata_loader, steps_per_epoch=None):
        self.m2ad_loader = m2ad_loader
        self.mydata_loader = mydata_loader
        if steps_per_epoch is None:
            steps_per_epoch = max(len(m2ad_loader), len(mydata_loader))
        self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        m_it = iter(self.m2ad_loader)
        y_it = iter(self.mydata_loader)
        for _ in range(self.steps_per_epoch):
            try:
                bm = next(m_it)
            except StopIteration:
                m_it = iter(self.m2ad_loader)
                bm = next(m_it)
            try:
                by = next(y_it)
            except StopIteration:
                y_it = iter(self.mydata_loader)
                by = next(y_it)
            yield {"m2ad": bm, "mydata": by}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _pool_mask(mask, output_shape):
    return F.adaptive_avg_pool2d(mask.unsqueeze(1).float(), output_shape).squeeze(1)


def _pair_positive_loss(feat_n1, feat_n2, fg_binary, margin_pos):
    """L = mean over FG of ReLU(β − cos(n1, n2))."""
    if not fg_binary.any():
        zero = feat_n1.new_tensor(0.0)
        return zero, {"pos_sim": 0.0, "n_fg": 0, "loss": 0.0}
    pos_sim_map = (feat_n1 * feat_n2).sum(dim=1)   # (B, h, w)
    pos = pos_sim_map[fg_binary]
    loss = F.relu(margin_pos - pos).mean()
    return loss, {
        "pos_sim": float(pos.mean().item()),
        "n_fg": int(fg_binary.sum().item()),
        "loss": float(loss.item()),
    }


def _triplet_margin_loss(
    feat_n1, feat_n2, feat_a1, fg_binary, def_labels, cfg
):
    """L = weighted combination of clean pull + defect triplet margin."""
    margin_trip = float(cfg.get("margin-triplet", 0.3))
    margin_pos = float(cfg.get("margin-positive", 0.95))
    w_clean = float(cfg.get("mydata-clean-weight", 1.0))
    w_defect = float(cfg.get("mydata-defect-weight", 3.0))

    pos_sim_map = (feat_n1 * feat_n2).sum(dim=1)
    hard_sim_map = (feat_n1 * feat_a1).sum(dim=1)

    fg_valid = fg_binary & (def_labels != -1)
    clean_mask = fg_valid & (def_labels == 0)
    defect_mask = fg_valid & (def_labels == 1)

    if clean_mask.any():
        loss_clean = F.relu(margin_pos - pos_sim_map[clean_mask]).mean()
    else:
        loss_clean = feat_n1.new_tensor(0.0)

    if defect_mask.any():
        gap = pos_sim_map[defect_mask] - hard_sim_map[defect_mask]
        loss_defect = F.relu(margin_trip - gap).mean()
    else:
        loss_defect = feat_n1.new_tensor(0.0)

    w_sum = max(w_clean + w_defect, 1e-8)
    loss = (w_clean * loss_clean + w_defect * loss_defect) / w_sum

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
            "loss": float(loss.item()),
        }
    return loss, metrics


def compute_dual_source_loss(model, dual_batch, cfg):
    """One forward pass through the backbone on all images from both sources,
    then split features and compute per-source losses, combined by config
    weights."""
    bm = dual_batch["m2ad"]
    by = dual_batch["mydata"]

    Bm = bm["n1"].size(0)
    By = by["n1"].size(0)

    all_imgs = torch.cat([
        bm["n1"], bm["n2"],
        by["n1"], by["n2"], by["a1"],
    ], dim=0).to(_device, non_blocking=True)

    m_fg = bm["fg_mask"].to(_device, non_blocking=True)
    y_fg = by["fg_mask"].to(_device, non_blocking=True)
    y_def = by["defect_mask"].to(_device, non_blocking=True)

    encode = (
        model.module.encode_single
        if isinstance(model, torch.nn.DataParallel)
        else model.encode_single
    )
    all_feats = encode(all_imgs)  # list over layers, each (Bm*2+By*3, C, h, w)

    num_layers = len(all_feats)
    layer_weights = cfg.get("layer-loss-weights") or [1.0] * num_layers
    wt = torch.tensor(layer_weights[:num_layers], dtype=torch.float32, device=_device)
    wt = wt / wt.sum().clamp_min(1e-8)

    fg_thresh = float(cfg["foreground-thresh"])
    clean_thresh = float(cfg["patch-clean-thresh"])
    defect_thresh = float(cfg["patch-defect-thresh"])
    margin_pos = float(cfg.get("margin-positive", 0.95))
    w_src_m2ad = float(cfg.get("m2ad-source-weight", 0.5))
    w_src_mydata = float(cfg.get("mydata-source-weight", 1.0))
    w_src_sum = max(w_src_m2ad + w_src_mydata, 1e-8)

    total_loss = torch.tensor(0.0, device=_device)
    agg = _zero_metrics()

    for li, feat in enumerate(all_feats):
        h, w_feat = feat.shape[-2:]

        # Split features back into per-source groups
        m_n1 = feat[:Bm]
        m_n2 = feat[Bm : 2 * Bm]
        y_n1 = feat[2 * Bm : 2 * Bm + By]
        y_n2 = feat[2 * Bm + By : 2 * Bm + 2 * By]
        y_a1 = feat[2 * Bm + 2 * By :]

        # M2AD pair: positive pull on FG patches
        m_fg_p = _pool_mask(m_fg, (h, w_feat)) > fg_thresh
        loss_m, met_m = _pair_positive_loss(m_n1, m_n2, m_fg_p, margin_pos)

        # mydata triplet: clean + defect
        y_fg_p = _pool_mask(y_fg, (h, w_feat)) > fg_thresh
        y_def_p = _pool_mask(y_def, (h, w_feat))
        y_def_labels = torch.full(y_def_p.shape, -1, dtype=torch.long, device=_device)
        y_def_labels[(y_def_p < clean_thresh) & y_fg_p] = 0
        y_def_labels[(y_def_p > defect_thresh) & y_fg_p] = 1
        loss_y, met_y = _triplet_margin_loss(
            y_n1, y_n2, y_a1, y_fg_p, y_def_labels, cfg
        )

        layer_loss = (w_src_m2ad * loss_m + w_src_mydata * loss_y) / w_src_sum
        total_loss = total_loss + wt[li] * layer_loss

        # Metric accumulation (layer-weighted)
        w_py = float(wt[li].item())
        agg["m2ad_pos_sim"] += w_py * met_m["pos_sim"]
        agg["m2ad_loss"] += w_py * met_m["loss"]
        agg["mydata_pos_sim"] += w_py * met_y["pos_sim"]
        agg["mydata_hard_neg_sim"] += w_py * met_y["hard_neg_sim"]
        agg["mydata_gap"] += w_py * met_y["gap"]
        agg["mydata_loss_clean"] += w_py * met_y["loss_clean"]
        agg["mydata_loss_defect"] += w_py * met_y["loss_defect"]
        agg["n_m2ad_fg"] = max(agg["n_m2ad_fg"], met_m["n_fg"])
        agg["n_mydata_clean"] = max(agg["n_mydata_clean"], met_y["n_clean"])
        agg["n_mydata_defect"] = max(agg["n_mydata_defect"], met_y["n_defect"])

    return total_loss, agg


def _zero_metrics():
    return {
        "m2ad_pos_sim": 0.0,
        "m2ad_loss": 0.0,
        "mydata_pos_sim": 0.0,
        "mydata_hard_neg_sim": 0.0,
        "mydata_gap": 0.0,
        "mydata_loss_clean": 0.0,
        "mydata_loss_defect": 0.0,
        "n_m2ad_fg": 0,
        "n_mydata_clean": 0,
        "n_mydata_defect": 0,
    }


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

    for dual_batch in bar:
        optimizer.zero_grad()
        loss, report = compute_dual_source_loss(model, dual_batch, cfg)

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

        lr = optimizer.param_groups[0]["lr"]
        mean_loss = sum(losses) / max(len(losses), 1)
        bar.set_postfix(
            lr=f"{lr:.2e}",
            loss=f"{loss.item():.4f}",
            avg=f"{mean_loss:.4f}",
            m_pos=f"{report['m2ad_pos_sim']:.3f}",
            y_pos=f"{report['mydata_pos_sim']:.3f}",
            y_gap=f"{report['mydata_gap']:.3f}",
            nd=f"{report['n_mydata_defect']}",
        )

        if _dry:
            break

    bar.close()

    keys_mean = (
        "m2ad_pos_sim", "m2ad_loss",
        "mydata_pos_sim", "mydata_hard_neg_sim", "mydata_gap",
        "mydata_loss_clean", "mydata_loss_defect",
    )
    mean_report = {
        k: sum(r[k] for r in reports) / max(len(reports), 1) for k in keys_mean
    }
    for k in ("n_m2ad_fg", "n_mydata_clean", "n_mydata_defect"):
        mean_report[k] = sum(r[k] for r in reports) / max(len(reports), 1)

    train_time = bar.format_dict.get("elapsed", 0.0)
    return sum(losses) / max(len(losses), 1), mean_report, train_time


# ---------------------------------------------------------------------------
# LR scheduler
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
    ds["mydata-root"] = resolve_project_path(ds["mydata-root"])

    out = args["wandb"].get("output-path")
    args["wandb"]["output-path"] = resolve_project_path(out) if out else get_output_path_by_utc()
    return args


def _build_loaders(ds_cfg):
    figsize = tuple(ds_cfg["figsize"])
    nw = int(ds_cfg.get("num-workers", 2))

    m2ad_loader = build_m2ad_pair_loader(
        m2ad_root=ds_cfg["m2ad-root"],
        json_path=ds_cfg["json-path"],
        mask_root=ds_cfg.get("mask-root"),
        split=ds_cfg.get("split", "train"),
        figsize=figsize,
        batch_size=int(ds_cfg["m2ad-per-batch"]),
        num_workers=nw,
        min_lights_per_view=ds_cfg.get("min-lights-per-view", 2),
        object_id_offset=0,
    )
    mydata_ds = MyDataTripletDataset(
        mydata_root=ds_cfg["mydata-root"],
        figsize=figsize,
        object_id_offset=100000,
        spatial_scale=tuple(ds_cfg.get("mydata-spatial-scale", [0.8, 1.0])),
    )
    mydata_loader = DataLoader(
        mydata_ds,
        batch_size=int(ds_cfg["mydata-per-batch"]),
        num_workers=nw,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    return DualSourceLoader(
        m2ad_loader, mydata_loader,
        steps_per_epoch=ds_cfg.get("steps-per-epoch"),
    )


def main(args):
    utils_torch.seed_everything(_seed, verbose=_verbose)

    # Model
    model = models.get_model(**args["model"])
    model = models.wrap_model_for_gpus(model, device=_device, gpu_ids=_gpu_ids)

    loader = _build_loaders(args["dataset"])
    xprint(f"DualSourceLoader: {len(loader)} steps per epoch")

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
            model, optimizer, scaler, loader,
            args["margin-loss"], epoch, int(opt_cfg["epochs"]),
        )
        scheduler.step()

        logs = {
            "loss": loss,
            "epoch": epoch,
            "time.train": train_time,
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
            "m2ad/pos_sim": report["m2ad_pos_sim"],
            "m2ad/loss": report["m2ad_loss"],
            "mydata/pos_sim": report["mydata_pos_sim"],
            "mydata/hard_neg_sim": report["mydata_hard_neg_sim"],
            "mydata/gap": report["mydata_gap"],
            "mydata/loss_clean": report["mydata_loss_clean"],
            "mydata/loss_defect": report["mydata_loss_defect"],
            "n_m2ad_fg": report["n_m2ad_fg"],
            "n_mydata_clean": report["n_mydata_clean"],
            "n_mydata_defect": report["n_mydata_defect"],
            "time/train": train_time,
            "lr": optimizer.param_groups[0]["lr"],
        })

        epoch_bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss:.4f}",
            m_pos=f"{report['m2ad_pos_sim']:.3f}",
            y_pos=f"{report['mydata_pos_sim']:.3f}",
            y_gap=f"{report['mydata_gap']:.3f}",
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
