"""
Training script: M2AD (synthetic anomaly) + mydata (real anomaly) with
direct pair margin loss, no in-batch cross-object negatives.

Three design changes relative to ``train_m2ad.py``:

1.  **Mixed batches** — each step pulls ``m2ad-per-batch`` triplets from the
    M2AD loader and ``mydata-per-batch`` triplets from the mydata loader,
    concatenated into a single batch. mydata triplets use the real GT
    defect masks shipped with the dataset.

2.  **Margin loss** — replaces InfoNCE. For each foreground anchor patch i
    (pooled down to the feature-map resolution):

        defect anchor :  L_i = max(0, α − (pos_sim_i − hard_neg_sim_i))
        clean  anchor :  L_i = max(0, β − pos_sim_i)

    where ``α`` is ``margin-triplet`` and ``β`` is ``margin-positive``.
    Clean/defect losses are averaged separately, then combined with the
    configured weights.

3.  **No cross-object in-batch negatives** — the anchor only sees its own
    N2 positive and A1 hard-negative, nothing from other objects in the
    batch. The ``object_idx`` field is kept in the dict for future use.

See Training_Program.md for motivation.
"""

import argparse
import itertools
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
from robust_defect_detection.datasets.mydata_triplet import (
    MyDataTripletDataset,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Global runtime config
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
# Combined loader: pairs M2AD + mydata batches on every step
# ---------------------------------------------------------------------------

class CombinedTripletLoader:
    """Wraps two DataLoaders; each __next__ yields a dict of concatenated
    tensors (N1, N2, A1, masks) along the batch dim, length = M2AD len +
    mydata len.

    The loader with the longer stream drives the iteration count;
    the shorter stream is cycled.
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
        m2ad_it = iter(self.m2ad_loader)
        mydata_it = iter(self.mydata_loader)
        for _ in range(self.steps_per_epoch):
            try:
                bm = next(m2ad_it)
            except StopIteration:
                m2ad_it = iter(self.m2ad_loader)
                bm = next(m2ad_it)
            try:
                by = next(mydata_it)
            except StopIteration:
                mydata_it = iter(self.mydata_loader)
                by = next(mydata_it)
            yield self._concat(bm, by)

    @staticmethod
    def _concat(a, b):
        out = {}
        for k in a:
            out[k] = torch.cat([a[k], b[k]], dim=0)
        return out


# ---------------------------------------------------------------------------
# Pair-margin loss (no cross-object negatives)
# ---------------------------------------------------------------------------

def _pool_mask(mask, output_shape):
    """Avg-pool a (B, H, W) float mask down to (B, h, w)."""
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
    One-layer pair-margin loss. No in-batch negatives.

    feat_* are (B, C, h, w), L2-normalised along C.
    fg_binary  : (B, h, w) bool
    def_labels : (B, h, w) int64 — 0 clean / 1 defect / -1 ignore
    """
    margin_trip = float(cfg.get("margin-triplet", 0.3))
    margin_pos = float(cfg.get("margin-positive", 0.95))
    w_clean = float(cfg.get("clean-weight", 1.0))
    w_defect = float(cfg.get("defect-weight", 3.0))

    # Per-patch cosine similarities (features are L2-normalised on C)
    pos_sim_map = (feat_n1 * feat_n2).sum(dim=1)   # (B, h, w)
    hard_sim_map = (feat_n1 * feat_a1).sum(dim=1)  # (B, h, w)

    fg_valid = fg_binary & (def_labels != -1)
    clean_mask = fg_valid & (def_labels == 0)
    defect_mask = fg_valid & (def_labels == 1)

    # Clean anchors: positive pull toward margin_pos
    if clean_mask.any():
        pos_clean = pos_sim_map[clean_mask]
        loss_clean = F.relu(margin_pos - pos_clean).mean()
    else:
        loss_clean = feat_n1.new_tensor(0.0)

    # Defect anchors: triplet margin on (pos − hard_neg)
    if defect_mask.any():
        pos_def = pos_sim_map[defect_mask]
        hard_def = hard_sim_map[defect_mask]
        gap = pos_def - hard_def
        loss_defect = F.relu(margin_trip - gap).mean()
    else:
        loss_defect = feat_n1.new_tensor(0.0)

    # Weighted combination — normalise weights to sum to 1 so the scale
    # is independent of how the user picks w_clean / w_defect.
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
        }
    return loss, metrics


def compute_margin_loss(model, batch, cfg):
    n1 = batch["n1"].to(_device, non_blocking=True)
    n2 = batch["n2"].to(_device, non_blocking=True)
    a1 = batch["a1"].to(_device, non_blocking=True)
    fg_mask = batch["fg_mask"].to(_device, non_blocking=True)
    defect_mask = batch["defect_mask"].to(_device, non_blocking=True)

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
    agg = _zero_metrics()

    for li, (fn1, fn2, fa1) in enumerate(zip(z_n1, z_n2, z_a1)):
        h, w_feat = fn1.shape[-2:]
        fg_p = _pool_mask(fg_mask, (h, w_feat)) > fg_thresh
        def_p = _pool_mask(defect_mask, (h, w_feat))

        def_labels = torch.full(def_p.shape, -1, dtype=torch.long, device=_device)
        def_labels[(def_p < clean_thresh) & fg_p] = 0
        def_labels[(def_p > defect_thresh) & fg_p] = 1

        layer_loss, metrics = _compute_layer_margin_loss(
            fn1, fn2, fa1, fg_p, def_labels, cfg
        )
        total_loss = total_loss + wt[li] * layer_loss

        w_py = float(wt[li].item())
        for k in ("pos_sim", "hard_neg_sim", "gap", "loss_clean", "loss_defect"):
            agg[k] += w_py * metrics[k]
        agg["n_clean"] = max(agg["n_clean"], metrics["n_clean"])
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

        lr = optimizer.param_groups[0]["lr"]
        mean_loss = sum(losses) / max(len(losses), 1)
        bar.set_postfix(
            lr=f"{lr:.2e}",
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

    keys_mean = ("pos_sim", "hard_neg_sim", "gap", "loss_clean", "loss_defect")
    mean_report = {
        k: sum(r[k] for r in reports) / max(len(reports), 1) for k in keys_mean
    }
    mean_report["n_clean"] = sum(r["n_clean"] for r in reports) / max(len(reports), 1)
    mean_report["n_defect"] = sum(r["n_defect"] for r in reports) / max(len(reports), 1)

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
    if ds.get("dtd-root"):
        ds["dtd-root"] = resolve_project_path(ds["dtd-root"])
    ds["mydata-root"] = resolve_project_path(ds["mydata-root"])

    out = args["wandb"].get("output-path")
    args["wandb"]["output-path"] = resolve_project_path(out) if out else get_output_path_by_utc()
    return args


def _build_loaders(ds_cfg):
    figsize = tuple(ds_cfg["figsize"])
    m2ad_bs = int(ds_cfg["m2ad-per-batch"])
    mydata_bs = int(ds_cfg["mydata-per-batch"])
    nw = int(ds_cfg.get("num-workers", 2))

    m2ad_loader = build_m2ad_triplet_loader(
        m2ad_root=ds_cfg["m2ad-root"],
        json_path=ds_cfg["json-path"],
        mask_root=ds_cfg.get("mask-root"),
        dtd_root=ds_cfg.get("dtd-root"),
        split=ds_cfg.get("split", "train"),
        figsize=figsize,
        batch_size=m2ad_bs,
        num_workers=nw,
        min_lights_per_view=ds_cfg.get("min-lights-per-view", 2),
        perlin_octaves=tuple(ds_cfg.get("perlin-octaves", [4, 6])),
        perlin_scale=tuple(ds_cfg.get("perlin-scale", [2.0, 6.0])),
        min_area_ratio=ds_cfg.get("min-area-ratio", 0.05),
        max_area_ratio=ds_cfg.get("max-area-ratio", 0.30),
    )

    # Large object_id_offset keeps mydata IDs disjoint from M2AD IDs
    # (even though the loss no longer uses in-batch negatives, we keep
    # object_idx meaningful for possible downstream analysis).
    mydata_ds = MyDataTripletDataset(
        mydata_root=ds_cfg["mydata-root"],
        figsize=figsize,
        object_id_offset=100000,
        spatial_scale=tuple(ds_cfg.get("mydata-spatial-scale", [0.8, 1.0])),
    )
    mydata_loader = DataLoader(
        mydata_ds,
        batch_size=mydata_bs,
        num_workers=nw,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    return CombinedTripletLoader(
        m2ad_loader, mydata_loader,
        steps_per_epoch=ds_cfg.get("steps-per-epoch"),
    )


def main(args):
    utils_torch.seed_everything(_seed, verbose=_verbose)

    # Model
    model = models.get_model(**args["model"])
    model = models.wrap_model_for_gpus(model, device=_device, gpu_ids=_gpu_ids)

    # Data
    loader = _build_loaders(args["dataset"])
    xprint(f"Combined loader: {len(loader)} steps per epoch")

    # Optimizer
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
            "loss/clean": report["loss_clean"],
            "loss/defect": report["loss_defect"],
            "margin/pos_sim": report["pos_sim"],
            "margin/hard_neg_sim": report["hard_neg_sim"],
            "margin/gap": report["gap"],
            "n_clean": report["n_clean"],
            "n_defect": report["n_defect"],
            "time/train": train_time,
            "lr": optimizer.param_groups[0]["lr"],
        })

        epoch_bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss:.4f}",
            pos=f"{report['pos_sim']:.3f}",
            hn=f"{report['hard_neg_sim']:.3f}",
            gap=f"{report['gap']:.3f}",
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
