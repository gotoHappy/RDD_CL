"""Bidirectional cross-attention anomaly detector.

Architecture (inspired by Robust-Scene-Change-Detection's
``dino2 + cross_attention`` model, with the parameter-passing bug fixed):

    img_ref  ─┐
              ├─► [frozen contrastive feature extractor] ─► f_r, f_q (B, C, h, w)
    img_query ┘                                                │
                                                                ▼
                                       ┌──── flatten to (B, N, C) sequences ───┐
                                       │                                       │
                       ┌─────────► TwoCrossAttention ─────────►┐               │
                       │   - x_new = x.attends_to(y)           │  × num_blocks │
                       │   - y_new = y.attends_to(x)           │               │
                       └────────────────────────────────────────┘               │
                                                                                │
                                          reshape back to (B, C, h, w)          │
                                                                                ▼
                                                  concat → conv 2C→C → conv C→1
                                                                                │
                                          bilinear upsample to target_shp        │
                                                                                ▼
                                                    logits (B, H, W) — anomaly score

The feature extractor is the trained ``MultiLayerContrastiveModel`` (DINOv3 +
LoRA + projector). Its weights stay frozen here — this stage only trains the
cross-attention blocks and the conv decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Single-direction cross attention with post-LN + FFN.

    Each Q token attends to all K/V tokens (which come from the *other*
    image), then a residual add + LayerNorm + FFN + residual + LayerNorm.
    """

    def __init__(self, embed_dim, num_heads=4, ffn_ratio=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        ffn_dim = int(embed_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v):
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        x = self.norm1(q + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class TwoCrossAttention(nn.Module):
    """Bidirectional cross-attention block.

    Holds two independent ``CrossAttentionBlock`` instances:
        * x_new = cross_attn(x as Q, y as K/V) — img_1 attends to img_2
        * y_new = cross_attn(y as Q, x as K/V) — img_2 attends to img_1

    Returns ``(x_new, y_new)``. Both tensors are sequence-shaped
    ``(B, N, C)`` and are augmented with information from the other image
    at semantically corresponding positions.
    """

    def __init__(self, embed_dim, num_heads=4, ffn_ratio=4, dropout=0.0):
        super().__init__()
        self.x_attends_y = CrossAttentionBlock(embed_dim, num_heads, ffn_ratio, dropout)
        self.y_attends_x = CrossAttentionBlock(embed_dim, num_heads, ffn_ratio, dropout)

    def forward(self, x, y):
        x_new = self.x_attends_y(x, y, y)
        y_new = self.y_attends_x(y, x, x)
        return x_new, y_new


class CrossAttentionAnomalyDetector(nn.Module):
    """Top-level anomaly detector.

    Inputs
    ------
    img_ref, img_query : (B, 3, H, W)

    Outputs
    -------
    logits : (B, H, W) — pre-sigmoid anomaly score map at full resolution
        Apply ``torch.sigmoid`` to obtain a probability map. The raw logits
        are also useful as anomaly scores directly (any monotonic post-
        processing preserves AUROC).

    Frozen
    ------
    The wrapped ``feature_extractor`` (a trained MultiLayerContrastiveModel)
    has all parameters frozen by this class. Only ``cross_blocks`` and
    ``decoder`` are trainable.
    """

    def __init__(
        self,
        feature_extractor,
        feature_layer_idx: int = -1,
        embed_dim: int | None = None,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        target_shp=(512, 512),
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()

        self.feature_layer_idx = int(feature_layer_idx)
        self.target_shp = tuple(target_shp)

        if embed_dim is None:
            embed_dim = self._infer_embed_dim()
        self.embed_dim = int(embed_dim)

        self.cross_blocks = nn.ModuleList([
            TwoCrossAttention(self.embed_dim, num_heads, ffn_ratio, dropout)
            for _ in range(int(num_blocks))
        ])

        self.decoder = nn.Sequential(
            nn.Conv2d(2 * self.embed_dim, self.embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, 1, kernel_size=1),
        )

    def _infer_embed_dim(self):
        """Dimensionality of one patch feature emitted by the projector."""
        proj = self.feature_extractor.projectors[0]
        if hasattr(proj, "fc2"):       # PatchProjector (MLP)
            return proj.fc2.out_features
        if hasattr(proj, "fc"):        # LinearPatchProjector
            return proj.fc.out_features
        # IdentityProjector → backbone output dim
        return self.feature_extractor.backbone.dino_1.num_features

    def train(self, mode=True):
        """Override so that ``model.train()`` keeps the frozen extractor
        in eval mode (avoid dropout/normalisation drift in DINO blocks).
        """
        super().train(mode)
        self.feature_extractor.eval()
        return self

    # ------------------------------------------------------------------

    def forward(self, img_ref, img_query):
        # Frozen feature extraction
        with torch.no_grad():
            ref_feats, qry_feats = self.feature_extractor.encode_pair(img_ref, img_query)
        x = ref_feats[self.feature_layer_idx]    # (B, C, h, w)
        y = qry_feats[self.feature_layer_idx]
        B, C, h, w = x.shape

        # Spatial → sequence
        x_seq = x.flatten(2).permute(0, 2, 1)    # (B, h*w, C)
        y_seq = y.flatten(2).permute(0, 2, 1)

        for block in self.cross_blocks:
            x_seq, y_seq = block(x_seq, y_seq)

        # Sequence → spatial
        x = x_seq.permute(0, 2, 1).reshape(B, C, h, w)
        y = y_seq.permute(0, 2, 1).reshape(B, C, h, w)

        # Concat both interaction-augmented features and decode to one channel
        z = torch.cat([x, y], dim=1)             # (B, 2C, h, w)
        z = self.decoder(z)                      # (B, 1, h, w)
        z = F.interpolate(z, size=self.target_shp, mode="bilinear", align_corners=False)
        return z.squeeze(1)                      # (B, H, W) logits


class MultiLayerCrossAttentionAnomalyDetector(nn.Module):
    """Multi-layer variant with learnable cross-layer fusion.

    Each selected extractor layer gets its own bidirectional cross-attention
    stack. The interaction-augmented per-layer features are concatenated
    channel-wise and fused by a lightweight 1x1-conv decoder, letting the
    model learn how much to use each backbone layer instead of averaging
    logits by hand.
    """

    def __init__(
        self,
        feature_extractor,
        feature_layers,
        embed_dim: int | None = None,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        target_shp=(512, 512),
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()

        self.extractor_layers = [int(layer) for layer in self.feature_extractor.layers]
        self.feature_layers = self._resolve_feature_layers(feature_layers)
        self.feature_layer_indices = [
            self.extractor_layers.index(layer) for layer in self.feature_layers
        ]
        self.target_shp = tuple(target_shp)

        if embed_dim is None:
            embed_dim = self._infer_embed_dim()
        self.embed_dim = int(embed_dim)

        self.layer_cross_blocks = nn.ModuleList([
            nn.ModuleList([
                TwoCrossAttention(self.embed_dim, num_heads, ffn_ratio, dropout)
                for _ in range(int(num_blocks))
            ])
            for _ in self.feature_layers
        ])

        n_layers = len(self.feature_layers)
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * self.embed_dim * n_layers, self.embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, 1, kernel_size=1),
        )

    def _resolve_feature_layers(self, feature_layers):
        if feature_layers is None:
            feature_layers = [self.extractor_layers[-1]]
        selected = [int(layer) for layer in feature_layers]
        if not selected:
            raise ValueError("feature-layers must contain at least one layer")
        if len(set(selected)) != len(selected):
            raise ValueError(f"feature-layers contains duplicates: {selected}")
        invalid = [layer for layer in selected if layer not in self.extractor_layers]
        if invalid:
            raise ValueError(
                f"feature-layers contains invalid layers {invalid}. "
                f"Available extractor layers are {self.extractor_layers}."
            )
        return selected

    def _infer_embed_dim(self):
        """Dimensionality of one patch feature emitted by the projector."""
        proj = self.feature_extractor.projectors[0]
        if hasattr(proj, "fc2"):
            return proj.fc2.out_features
        if hasattr(proj, "fc"):
            return proj.fc.out_features
        return self.feature_extractor.backbone.dino_1.num_features

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    # ------------------------------------------------------------------

    def forward(self, img_ref, img_query):
        with torch.no_grad():
            ref_feats, qry_feats = self.feature_extractor.encode_pair(img_ref, img_query)

        fused_inputs = []
        spatial_hw = None

        for layer_idx, blocks in zip(self.feature_layer_indices, self.layer_cross_blocks):
            x = ref_feats[layer_idx]
            y = qry_feats[layer_idx]
            B, C, h, w = x.shape
            if spatial_hw is None:
                spatial_hw = (h, w)
            elif spatial_hw != (h, w):
                raise RuntimeError(
                    "selected feature layers have different spatial sizes: "
                    f"{spatial_hw} vs {(h, w)}"
                )

            x_seq = x.flatten(2).permute(0, 2, 1)
            y_seq = y.flatten(2).permute(0, 2, 1)

            for block in blocks:
                x_seq, y_seq = block(x_seq, y_seq)

            x = x_seq.permute(0, 2, 1).reshape(B, C, h, w)
            y = y_seq.permute(0, 2, 1).reshape(B, C, h, w)
            fused_inputs.append(torch.cat([x, y], dim=1))

        z = torch.cat(fused_inputs, dim=1)
        z = self.decoder(z)
        z = F.interpolate(z, size=self.target_shp, mode="bilinear", align_corners=False)
        return z.squeeze(1)
