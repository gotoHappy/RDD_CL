import torch
import torch.nn as nn

from .. import utils_torch
from .backbone_dinov2 import get_dino
from .backbone_dinov3 import get_dino_v3
from .contrastive import MultiLayerContrastiveModel


def _build_backbone(opts):
    name = opts.get("dino-model", "dinov2_vits14")
    freeze = opts.get("freeze-dino", True)
    unfreeze_last = opts.get("unfreeze-dino-last-n-layer", 0)
    if name.startswith("dinov3"):
        return get_dino_v3(
            name,
            freeze=freeze,
            unfreeze_last_n_layers=unfreeze_last,
            weights_path=opts.get("dino-weights-path"),
        )
    return get_dino(name, freeze=freeze, unfreeze_last_n_layers=unfreeze_last)


def get_model(**opts):
    name = opts.get("name")
    if name == "dino2 + contrast_learning":
        return _build_contrast_model(opts)
    raise ValueError(f"unsupported model: {name}")


def _build_contrast_model(opts):
    backbone = _build_backbone(opts)
    target_shp = (
        int(opts["target-shp-row"]),
        int(opts["target-shp-col"]),
    )
    return MultiLayerContrastiveModel(
        backbone,
        layers=opts.get("layers"),
        proj_hidden_dim=opts.get("proj-hidden-dim", 512),
        proj_out_dim=opts.get("proj-out-dim", 256),
        proj_use_layer_norm=opts.get("proj-use-layer-norm", True),
        projector_type=opts.get("projector-type", "mlp"),
        target_shp=target_shp,
    )


def wrap_model_for_gpus(model, device="cuda", gpu_ids=None):
    model = model.to(device)
    if str(device).startswith("cuda") and gpu_ids is not None and len(gpu_ids) > 1:
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model


def load_checkpoint_model(path, device="cuda", gpu_ids=None, verbose=False):
    checkpoint = torch.load(path, map_location="cpu")
    model_args = checkpoint["args"]["model"]
    name = model_args.get("name")

    if name == "dino2 + contrast_learning":
        model = _build_contrast_model(model_args)
        model = wrap_model_for_gpus(model, device=device, gpu_ids=gpu_ids)
        model = utils_torch.load_grad_required_state(model, checkpoint["model"], verbose=verbose)
        model.eval()
        return model, checkpoint

    raise ValueError(f"unsupported checkpoint model name: {name}")
