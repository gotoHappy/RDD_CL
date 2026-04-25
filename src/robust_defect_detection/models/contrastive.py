import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvff

from .backbone_dinov2 import TwoMultiLayerDino
from .backbone_dinov3 import TwoMultiLayerDinoV3


def _pick_two_multilayer_cls(dino):
    """DINOv3 has ``n_storage_tokens > 0`` register tokens that must be
    skipped when extracting patch tokens. DINOv2 does not. Pick the
    appropriate wrapper accordingly."""
    if int(getattr(dino, "n_storage_tokens", 0)) > 0:
        return TwoMultiLayerDinoV3
    return TwoMultiLayerDino


class PatchProjector(nn.Module):
    """MLP projector: Linear → GELU → [LayerNorm] → Linear → L2-normalise."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=256, use_layer_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        x = self.fc1(x)
        x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc2(x)
        x = F.normalize(x, dim=-1)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        return x


class LinearPatchProjector(nn.Module):
    """Lightweight projector: single ``Linear(in_dim → out_dim)`` + L2-norm.
    No hidden layer, no activation, no LayerNorm."""

    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        return x


class IdentityProjector(nn.Module):
    """No projection. Backbone features are already L2-normalised along the
    channel dimension by ``MultiLayerExtractDINOv3._forward``, so cosine
    distances on these features are directly meaningful."""

    def forward(self, x):
        return x


def _build_projector(projector_type, in_dim, hidden_dim, out_dim, use_layer_norm):
    if projector_type == "mlp":
        return PatchProjector(in_dim, hidden_dim, out_dim, use_layer_norm=use_layer_norm)
    if projector_type == "linear":
        return LinearPatchProjector(in_dim, out_dim)
    if projector_type == "identity":
        return IdentityProjector()
    raise ValueError(
        f"unknown projector_type: {projector_type!r} "
        "(expected one of 'mlp', 'linear', 'identity')"
    )


class MultiLayerContrastiveModel(nn.Module):
    def __init__(
        self,
        dino1,
        layers=None,
        dino2=None,
        proj_hidden_dim=512,
        proj_out_dim=256,
        proj_use_layer_norm=True,
        projector_type="mlp",
        target_shp=(504, 504),
        **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 5, 8, 11]
        dino2 = dino1 if dino2 is None else dino2
        self.layers = [int(layer) for layer in layers]
        two_ml_cls = _pick_two_multilayer_cls(dino1)
        self.backbone = two_ml_cls(
            dino_1=dino1,
            layers_1=self.layers,
            dino_2=dino2,
            layers_2=self.layers,
        )
        in_dim = self.backbone.dino_1.num_features
        self.projector_type = projector_type
        self.projectors = nn.ModuleList(
            [
                _build_projector(
                    projector_type, in_dim, proj_hidden_dim, proj_out_dim, proj_use_layer_norm
                )
                for _ in self.layers
            ]
        )
        self.target_shp = target_shp

    def encode_single(self, img):
        features = self.backbone.dino_1(img)
        return [projector(feature) for projector, feature in zip(self.projectors, features)]

    def encode_pair(self, img_ref, img_query):
        ref_features, query_features = self.backbone(img_ref, img_query)
        ref_features = [projector(feature) for projector, feature in zip(self.projectors, ref_features)]
        query_features = [projector(feature) for projector, feature in zip(self.projectors, query_features)]
        return ref_features, query_features

    @staticmethod
    def compute_layer_distance(ref_feature, query_feature):
        cosine = F.cosine_similarity(ref_feature, query_feature, dim=1)
        return 1.0 - cosine

    def compute_anomaly_maps(
        self,
        img_ref,
        img_query,
        layer_weights=None,
        smooth_kernel=5,
        smooth_sigma=1.0,
    ):
        ref_features, query_features = self.encode_pair(img_ref, img_query)
        layer_maps = []
        for ref_feature, query_feature in zip(ref_features, query_features):
            score = self.compute_layer_distance(ref_feature, query_feature)
            score = score.unsqueeze(1)
            score = F.interpolate(
                score,
                size=self.target_shp,
                mode="bilinear",
                align_corners=False,
            )
            layer_maps.append(score)

        if layer_weights is None:
            layer_weights = [float(i + 1) for i in range(len(layer_maps))]
        weights = torch.tensor(
            layer_weights,
            dtype=layer_maps[0].dtype,
            device=layer_maps[0].device,
        )
        weights = weights / weights.sum().clamp_min(1e-8)

        fused = 0.0
        for weight, layer_map in zip(weights, layer_maps):
            fused = fused + weight * layer_map

        if smooth_kernel and smooth_kernel > 1:
            fused = tvff.gaussian_blur(
                fused,
                kernel_size=[smooth_kernel, smooth_kernel],
                sigma=[smooth_sigma, smooth_sigma],
            )

        return fused.squeeze(1), [layer_map.squeeze(1) for layer_map in layer_maps]

    def compute_dino_baseline_maps(
        self,
        img_ref,
        img_query,
        layer_weights=None,
        smooth_kernel=5,
        smooth_sigma=1.0,
    ):
        ref_features, query_features = self.backbone(img_ref, img_query)
        layer_maps = []
        for ref_feature, query_feature in zip(ref_features, query_features):
            score = self.compute_layer_distance(ref_feature, query_feature)
            score = score.unsqueeze(1)
            score = F.interpolate(
                score,
                size=self.target_shp,
                mode="bilinear",
                align_corners=False,
            )
            layer_maps.append(score)

        if layer_weights is None:
            layer_weights = [float(i + 1) for i in range(len(layer_maps))]
        weights = torch.tensor(
            layer_weights,
            dtype=layer_maps[0].dtype,
            device=layer_maps[0].device,
        )
        weights = weights / weights.sum().clamp_min(1e-8)

        fused = 0.0
        for weight, layer_map in zip(weights, layer_maps):
            fused = fused + weight * layer_map

        if smooth_kernel and smooth_kernel > 1:
            fused = tvff.gaussian_blur(
                fused,
                kernel_size=[smooth_kernel, smooth_kernel],
                sigma=[smooth_sigma, smooth_sigma],
            )

        return fused.squeeze(1), [layer_map.squeeze(1) for layer_map in layer_maps]
