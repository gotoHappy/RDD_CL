import argparse
import json
import os
import pickle
import sys

import torch
import wandb
import yaml
from torch.nn import functional as F
from tqdm.auto import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import robust_defect_detection.datasets as datasets
import robust_defect_detection.models as models
from robust_defect_detection import utils, utils_torch


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


def pool_mask(mask, output_shape):
    return F.adaptive_avg_pool2d(mask.unsqueeze(1), output_shape).squeeze(1)


def masked_mean(values, mask):
    mask = mask.float()
    denom = mask.sum()
    if denom <= 0:
        return values.new_tensor(0.0), values.new_tensor(0.0)
    return (values * mask).sum() / denom, denom


def compute_contrastive_loss(model, batch, cfg):
    clean_thresh = cfg["patch-clean-thresh"]
    defect_thresh = cfg["patch-defect-thresh"]
    fg_thresh = cfg["foreground-thresh"]
    margin = cfg["margin"]
    lambda_nn = cfg["lambda-nn"]
    lambda_clean = cfg["lambda-clean"]
    lambda_defect = cfg["lambda-defect"]
    layer_weights = cfg.get("layer-loss-weights")

    normal_1 = batch["normal_1"].to(_device)
    normal_2 = batch["normal_2"].to(_device)
    anomaly = batch["anomaly"].to(_device)
    defect_mask = batch["defect_mask"].to(_device)
    fg_mask = batch["fg_mask"].to(_device)

    encode_single = model.module.encode_single if isinstance(model, torch.nn.DataParallel) else model.encode_single
    z_n1 = encode_single(normal_1)
    z_n2 = encode_single(normal_2)
    z_a1 = encode_single(anomaly)

    num_layers = len(z_n1)
    if layer_weights is None:
        layer_weights = [float(i + 1) for i in range(num_layers)]
    weight_tensor = torch.tensor(layer_weights, dtype=torch.float32, device=_device)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)

    total_loss = torch.tensor(0.0, device=_device)
    total_nn = torch.tensor(0.0, device=_device)
    total_clean = torch.tensor(0.0, device=_device)
    total_def = torch.tensor(0.0, device=_device)

    for weight, feat_n1, feat_n2, feat_a1 in zip(weight_tensor, z_n1, z_n2, z_a1):
        output_shape = feat_n1.shape[-2:]
        pooled_defect = pool_mask(defect_mask, output_shape)
        pooled_fg = pool_mask(fg_mask, output_shape)

        nn_mask = pooled_fg > fg_thresh
        clean_mask = (pooled_defect < clean_thresh) & nn_mask
        defect_valid_mask = (pooled_defect > defect_thresh) & nn_mask

        d_nn = 1.0 - F.cosine_similarity(feat_n1, feat_n2, dim=1)
        d_na = 1.0 - F.cosine_similarity(feat_n1, feat_a1, dim=1)

        loss_nn, _ = masked_mean(d_nn, nn_mask)
        loss_clean, _ = masked_mean(d_na, clean_mask)
        loss_def, _ = masked_mean(F.relu(margin - d_na), defect_valid_mask)

        layer_loss = lambda_nn * loss_nn + lambda_clean * loss_clean + lambda_defect * loss_def
        total_loss = total_loss + weight * layer_loss
        total_nn = total_nn + weight * loss_nn
        total_clean = total_clean + weight * loss_clean
        total_def = total_def + weight * loss_def

    return total_loss, {
        "loss_nn": float(total_nn.detach().cpu()),
        "loss_clean": float(total_clean.detach().cpu()),
        "loss_defect": float(total_def.detach().cpu()),
    }


def train_one_epoch(model, optimizer, scaler, loader, contrast_cfg, epoch_idx, total_epochs):
    model.train()
    losses = []
    reports = []

    inner_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch_idx + 1}/{total_epochs}",
        position=1,
        leave=False,
        dynamic_ncols=True,
        ascii=False,
    )

    for idx, batch in enumerate(inner_bar):
        optimizer.zero_grad()
        loss, report = compute_contrastive_loss(model, batch, contrast_cfg)
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
        lr = optimizer.param_groups[0]["lr"]
        inner_bar.set_postfix(
            lr=f"{lr:.2e}",
            loss=f"{loss.item():.4f}",
            avg=f"{mean_loss:.4f}",
            nn=f"{report['loss_nn']:.4f}",
            clean=f"{report['loss_clean']:.4f}",
            defect=f"{report['loss_defect']:.4f}",
        )
        if _dry:
            break

    inner_bar.close()

    mean_report = {
        "loss_nn": sum(r["loss_nn"] for r in reports) / max(len(reports), 1),
        "loss_clean": sum(r["loss_clean"] for r in reports) / max(len(reports), 1),
        "loss_defect": sum(r["loss_defect"] for r in reports) / max(len(reports), 1),
    }
    train_time = inner_bar.format_dict.get("elapsed", 0.0)
    return sum(losses) / max(len(losses), 1), mean_report, train_time


def get_output_path_by_utc():
    return os.path.join(_PROJECT_ROOT, "outputs", utils.get_utc_time())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parsed = parser.parse_args()
    with open(os.path.abspath(parsed.config), "r") as fd:
        args = yaml.safe_load(fd)
    args["dataset"]["root"] = resolve_project_path(args["dataset"]["root"])
    output_path = args["wandb"].get("output-path")
    args["wandb"]["output-path"] = resolve_project_path(output_path) if output_path else get_output_path_by_utc()
    return args


def main(args):
    utils_torch.seed_everything(_seed, verbose=_verbose)

    model = models.get_model(**args["model"])
    model = models.wrap_model_for_gpus(model, device=_device, gpu_ids=_gpu_ids)

    loader = datasets.build_contrastive_triplet_loader(
        root=args["dataset"]["root"],
        figsize=args["dataset"]["figsize"],
        batch_size=args["dataset"]["batch-size"],
        num_workers=args["dataset"]["num-workers"],
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args["optimizer"]["learn-rate"])
    scheduler = utils_torch.CustomizedLRScheduler(
        optimizer,
        start_scale=0.0,
        warmup_epoch=args["optimizer"]["warmup-epoch"],
        final_scale=0.2,
        total_epoch=args["optimizer"]["epochs"],
        mode=None if args["optimizer"]["lr-scheduler"].lower() == "none" else args["optimizer"]["lr-scheduler"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args["optimizer"]["grad-scaler"])

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
        range(args["optimizer"]["epochs"]),
        total=args["optimizer"]["epochs"],
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
            args["optimizer"]["epochs"],
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
                "loss/nn": report["loss_nn"],
                "loss/clean": report["loss_clean"],
                "loss/defect": report["loss_defect"],
                "time/train": train_time,
            }
        )

        epoch_bar.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            loss=f"{loss:.4f}",
            nn=f"{report['loss_nn']:.4f}",
            clean=f"{report['loss_clean']:.4f}",
            defect=f"{report['loss_defect']:.4f}",
            best=f"{best_loss:.4f}" if best_loss < float("inf") else "n/a",
        )

        save_freq = args["wandb"]["save-checkpoint-freq"]
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(checkpoint, os.path.join(args["wandb"]["output-path"], "checkpoints", f"{epoch}.layer.pth"))
        if loss < best_loss:
            best_loss = loss
            torch.save(checkpoint, os.path.join(args["wandb"]["output-path"], "best.pth"))
            epoch_bar.set_postfix(
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                loss=f"{loss:.4f}",
                nn=f"{report['loss_nn']:.4f}",
                clean=f"{report['loss_clean']:.4f}",
                defect=f"{report['loss_defect']:.4f}",
                best=f"{best_loss:.4f}",
            )
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
