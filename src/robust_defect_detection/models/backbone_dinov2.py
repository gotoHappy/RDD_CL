from pathlib import Path
import threading
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..utils_torch import freeze_model, is_all_frozen, is_any_frozen, unfreeze_model

warnings.filterwarnings("ignore")

_DEFAULT_CACHE_REPO = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"


def _get_dino(model="dinov2_vitg14", repo=None):
    repo = Path(repo) if repo is not None else _DEFAULT_CACHE_REPO
    if repo.exists():
        return torch.hub.load(str(repo), model, source="local")
    return torch.hub.load("facebookresearch/dinov2", model)


def get_dino(model, freeze=True, unfreeze_last_n_layers=0):
    dino_model = _get_dino(model)

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


class MultiLayerExtractDINO(nn.Module):
    def __init__(self, dino, layers):
        super().__init__()
        self.dino = dino
        self.num_features = dino.num_features
        self.patch_size = dino.patch_size
        self.embed_dim = dino.embed_dim
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
            token = outputs[layer][:, 1:, ...]
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


class TwoMultiLayerDino(nn.Module):
    def __init__(self, dino_1, layers_1, dino_2, layers_2):
        super().__init__()
        self.dino_1 = MultiLayerExtractDINO(dino_1, layers=layers_1)
        self.dino_2 = MultiLayerExtractDINO(dino_2, layers=layers_2)

    def forward(self, img_1, img_2):
        x1 = self.dino_1(img_1)
        x2 = self.dino_2(img_2)
        return x1, x2
