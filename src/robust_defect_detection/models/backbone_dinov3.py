"""DINOv3 backbone loader.

Loads DINOv3 ViT models from the local repository at
``/data2/baizeyu/dinov3``. The module's ``hubconf.py`` pulls in segmentation
/ detection code that requires extra packages (e.g. ``torchmetrics``), so we
skip ``torch.hub.load`` and import the backbone factory functions directly
— this only needs ``dinov3.hub.backbones``.

The key difference from DINOv2 is that DINOv3 ViT models use
``n_storage_tokens = 4`` register tokens. Each transformer block therefore
outputs ``[CLS, storage_0..storage_{K-1}, patch_0..patch_{N-1}]``. Patch
extraction in :class:`MultiLayerExtractDINOv3` skips the first
``1 + n_storage_tokens`` tokens accordingly.
"""

from pathlib import Path
import sys
import threading
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils_torch import freeze_model, is_all_frozen, is_any_frozen, unfreeze_model

warnings.filterwarnings("ignore")

DINOV3_REPO_PATH = Path("/data2/baizeyu/dinov3")


def _ensure_dinov3_on_path():
    if not DINOV3_REPO_PATH.exists():
        raise FileNotFoundError(
            f"DINOv3 repo not found at {DINOV3_REPO_PATH}. "
            "Clone https://github.com/facebookresearch/dinov3 there first."
        )
    repo_str = str(DINOV3_REPO_PATH)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _load_dinov3(model_name: str, weights_path=None):
    """Instantiate a DINOv3 backbone from the local repo.

    Parameters
    ----------
    model_name : str
        Backbone factory name, e.g. ``dinov3_vitb16``.
    weights_path : str | Path | None
        Path to a local ``*.pth`` file. When ``None`` the model is
        instantiated with ``pretrained=False`` (random init) — useful for
        smoke-testing before weights are available. A warning is emitted.
    """
    _ensure_dinov3_on_path()
    from dinov3.hub import backbones as _v3_backbones

    try:
        factory = getattr(_v3_backbones, model_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown DINOv3 backbone: {model_name}. "
            "Available names are exported from dinov3.hub.backbones."
        ) from exc

    if weights_path is None:
        warnings.warn(
            f"Loading {model_name} with random weights (pretrained=False). "
            "Download the official checkpoint and set `dino-weights-path` "
            "in the config before real training.",
            RuntimeWarning,
        )
        return factory(pretrained=False)

    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    return factory(pretrained=True, weights=str(weights_path))


def get_dino_v3(model, freeze=True, unfreeze_last_n_layers=0, weights_path=None):
    dino_model = _load_dinov3(model, weights_path=weights_path)

    if freeze:
        freeze_model(dino_model)
        assert is_all_frozen(dino_model)
    else:
        assert not is_any_frozen(dino_model)

    total_blocks = len(dino_model.blocks)
    split_idx = max(total_blocks - max(unfreeze_last_n_layers, 0), 0)
    for block in dino_model.blocks[split_idx:]:
        unfreeze_model(block)

    return dino_model


class MultiLayerExtractDINOv3(nn.Module):
    """Extract and L2-normalise patch tokens from selected DINOv3 blocks.

    The block output layout is ``[CLS, storage × K, patch × N]`` where
    ``K = dino.n_storage_tokens``. Patch tokens start at index ``K + 1``.
    """

    def __init__(self, dino, layers):
        super().__init__()
        self.dino = dino
        self.num_features = dino.num_features
        self.patch_size = dino.patch_size
        self.embed_dim = dino.embed_dim
        self.n_prefix = 1 + int(getattr(dino, "n_storage_tokens", 0))
        self.layers = [int(layer) for layer in layers]
        self._handles = []
        self._hook_output = {}
        self.frozen_mode = is_all_frozen(dino)

    def _del_handles(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _set_handles(self):
        def make_hook(layer):
            def hook(module, inputs, outputs):
                thread_id = threading.get_native_id()
                if thread_id not in self._hook_output:
                    self._hook_output[thread_id] = {}
                self._hook_output[thread_id][layer] = outputs

            return hook

        for layer in self.layers:
            handle = self.dino.blocks[layer].register_forward_hook(make_hook(layer))
            self._handles.append(handle)

    def _forward(self, x):
        self._set_handles()

        batch, _, row, col = x.shape
        _ = self.dino(x)

        thread_id = threading.get_native_id()
        outputs = self._hook_output.pop(thread_id)
        self._del_handles()

        features = []
        for layer in self.layers:
            raw = outputs[layer]
            # DINOv3 blocks always receive/return a list of tensors (see
            # forward_features_list); for a single-image forward the list
            # has length 1.
            if isinstance(raw, (list, tuple)):
                assert len(raw) == 1, f"unexpected block output list len: {len(raw)}"
                raw = raw[0]
            token = raw[:, self.n_prefix :, ...]
            token = F.normalize(token, dim=-1)
            new_shape = (batch, row // self.patch_size, col // self.patch_size, -1)
            token = token.reshape(new_shape).permute(0, 3, 1, 2)
            features.append(token)

        return features

    def forward(self, x):
        if not self.frozen_mode:
            return self._forward(x)

        if self.dino.training:
            self.dino.eval()

        with torch.no_grad():
            return self._forward(x)


class TwoMultiLayerDinoV3(nn.Module):
    def __init__(self, dino_1, layers_1, dino_2, layers_2):
        super().__init__()
        self.dino_1 = MultiLayerExtractDINOv3(dino_1, layers=layers_1)
        self.dino_2 = MultiLayerExtractDINOv3(dino_2, layers=layers_2)

    def forward(self, img_1, img_2):
        x1 = self.dino_1(img_1)
        x2 = self.dino_2(img_2)
        return x1, x2
