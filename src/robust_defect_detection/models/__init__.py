import torch
import torch.nn as nn

from .. import utils_torch
from .backbone_dinov2 import get_dino
from .backbone_dinov3 import get_dino_v3
from .contrastive import MultiLayerContrastiveModel
from .cross_attention import CrossAttentionAnomalyDetector, MultiLayerCrossAttentionAnomalyDetector


def _build_backbone(opts):
    name = opts.get("dino-model", "dinov2_vits14")
    freeze = opts.get("freeze-dino", True)
    unfreeze_last = opts.get("unfreeze-dino-last-n-layer", 0)
    if name.startswith("dinov3"):
        dino = get_dino_v3(
            name,
            freeze=freeze,
            unfreeze_last_n_layers=unfreeze_last,
            weights_path=opts.get("dino-weights-path"),
        )
    else:
        dino = get_dino(name, freeze=freeze, unfreeze_last_n_layers=unfreeze_last)

    # Optional LoRA injection (after the backbone is loaded and frozen).
    # Config schema:
    #   lora:
    #     rank: 16
    #     alpha: null                 # null → 2 * rank
    #     blocks: [5, 6, 7, 8, 9, 10, 11]
    lora_cfg = opts.get("lora")
    if lora_cfg:
        from .lora import inject_lora_attn

        block_indices = list(lora_cfg["blocks"])
        rank = int(lora_cfg.get("rank", 16))
        alpha = lora_cfg.get("alpha")
        alpha = float(alpha) if alpha is not None else None
        inject_lora_attn(dino, block_indices, rank=rank, alpha=alpha)

    return dino


def get_model(**opts):
    name = opts.get("name")
    if name == "dino2 + contrast_learning":
        return _build_contrast_model(opts)
    if name == "dino2 + cross_attention":
        return _build_cross_attention_model(opts)
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


def _build_cross_attention_model(opts):
    """Build a CrossAttentionAnomalyDetector on top of a previously-trained
    contrastive feature extractor.

    Required key:
        ``backbone-args``  — the saved ``args["model"]`` dict from the
                              contrastive checkpoint.  Used to rebuild the
                              extractor architecture (LoRA injection, projector
                              type, etc).  The extractor's *trainable* state
                              (LoRA matrices + projector weights) lives in
                              ``checkpoint["backbone_state"]`` and is loaded
                              by the train script / load_checkpoint_model.
    Optional keys:
        feature-layers      — real extractor layer IDs to feed cross-attention
                              (default: derived from feature-layer-idx)
        feature-layer-idx   — legacy list index of the extractor layer
                              (default ``-1`` = last)
        embed-dim           — override; default = projector output dim
        num-heads / num-blocks / ffn-ratio / dropout
        target-shp-row / target-shp-col — output mask resolution
    """
    backbone_args = opts.get("backbone-args")
    if not backbone_args:
        raise ValueError(
            "'backbone-args' (a dict mirroring the contrastive checkpoint's "
            "args['model']) is required for the cross_attention model."
        )
    feature_extractor = _build_contrast_model(backbone_args)
    target_shp = (
        int(opts["target-shp-row"]),
        int(opts["target-shp-col"]),
    )
    feature_layers = opts.get("feature-layers")
    if feature_layers is not None:
        return MultiLayerCrossAttentionAnomalyDetector(
            feature_extractor=feature_extractor,
            feature_layers=feature_layers,
            embed_dim=opts.get("embed-dim"),
            num_heads=int(opts.get("num-heads", 4)),
            num_blocks=int(opts.get("num-blocks", 2)),
            ffn_ratio=float(opts.get("ffn-ratio", 4.0)),
            dropout=float(opts.get("dropout", 0.0)),
            target_shp=target_shp,
        )
    return CrossAttentionAnomalyDetector(
        feature_extractor=feature_extractor,
        feature_layer_idx=int(opts.get("feature-layer-idx", -1)),
        embed_dim=opts.get("embed-dim"),
        num_heads=int(opts.get("num-heads", 4)),
        num_blocks=int(opts.get("num-blocks", 2)),
        ffn_ratio=float(opts.get("ffn-ratio", 4.0)),
        dropout=float(opts.get("dropout", 0.0)),
        target_shp=target_shp,
    )


def wrap_model_for_gpus(model, device="cuda", gpu_ids=None):
    model = model.to(device)
    if str(device).startswith("cuda") and gpu_ids is not None and len(gpu_ids) > 1:
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model


def _strip_module_prefix(state):
    out = {}
    for k, v in state.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


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

    if name == "dino2 + cross_attention":
        # 1. Rebuild the contrastive feature extractor (DINO + LoRA + projector
        #    *structure*, no trained weights yet)
        backbone_args = model_args["backbone-args"]
        extractor = _build_contrast_model(backbone_args)
        # 2. Restore the extractor's trained LoRA + projector weights from
        #    the cross-attention checkpoint's sidecar dict. We bypass the
        #    grad-skip logic and use plain ``load_state_dict(strict=False)``
        #    so the frozen-by-design LoRA/projector tensors actually get
        #    populated.
        ext_state = checkpoint.get("extractor_state")
        if ext_state:
            extractor.load_state_dict(_strip_module_prefix(ext_state), strict=False)
        # 3. Build cross-attention head on the now-loaded extractor
        target_shp = (
            int(model_args["target-shp-row"]),
            int(model_args["target-shp-col"]),
        )
        feature_layers = model_args.get("feature-layers")
        if feature_layers is not None:
            ca_model = MultiLayerCrossAttentionAnomalyDetector(
                feature_extractor=extractor,
                feature_layers=feature_layers,
                embed_dim=model_args.get("embed-dim"),
                num_heads=int(model_args.get("num-heads", 4)),
                num_blocks=int(model_args.get("num-blocks", 2)),
                ffn_ratio=float(model_args.get("ffn-ratio", 4.0)),
                dropout=float(model_args.get("dropout", 0.0)),
                target_shp=target_shp,
            )
        else:
            ca_model = CrossAttentionAnomalyDetector(
                feature_extractor=extractor,
                feature_layer_idx=int(model_args.get("feature-layer-idx", -1)),
                embed_dim=model_args.get("embed-dim"),
                num_heads=int(model_args.get("num-heads", 4)),
                num_blocks=int(model_args.get("num-blocks", 2)),
                ffn_ratio=float(model_args.get("ffn-ratio", 4.0)),
                dropout=float(model_args.get("dropout", 0.0)),
                target_shp=target_shp,
            )
        ca_model = wrap_model_for_gpus(ca_model, device=device, gpu_ids=gpu_ids)
        # 4. Load trained cross-attention + decoder weights
        ca_model = utils_torch.load_grad_required_state(
            ca_model, checkpoint["model"], verbose=verbose,
        )
        ca_model.eval()
        return ca_model, checkpoint

    raise ValueError(f"unsupported checkpoint model name: {name}")
