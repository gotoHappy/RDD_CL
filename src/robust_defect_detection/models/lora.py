"""LoRA injection for DINOv3 attention layers (qkv + proj only).

Wraps the original ``nn.Linear`` with a low-rank update::

    y = W(x) + (alpha / rank) * B(A(x))

where ``A`` is initialised with kaiming_uniform and ``B`` with zeros, so the
LoRA branch contributes zero at the start of training and the wrapped layer
behaves exactly like the frozen base. Only ``A`` and ``B`` are trainable;
the original ``linear`` is frozen.

DINOv3's combined QKV uses ``LinearKMaskedBias`` (a subclass of
``nn.Linear``); LoRA wrapping is safe because we always call the original
module via ``self.linear(x)``, preserving its bias-masking logic.
"""

import math

import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.scaling = float(alpha) / float(rank)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.linear.parameters():
            p.requires_grad = False

    # Mirror the public interface of ``nn.Linear`` so callers reading
    # metadata (e.g. ``self.qkv.in_features`` inside DINOv3's attention)
    # transparently see the wrapped layer's shape.
    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, x):
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(x))


def inject_lora_attn(dino, block_indices, rank=16, alpha=None):
    """Replace ``blocks[i].attn.qkv`` and ``blocks[i].attn.proj`` with
    :class:`LoRALinear` for every ``i`` in ``block_indices``.

    ``alpha`` defaults to ``2 * rank`` (so ``scaling = 2.0``).
    Returns the modified ``dino`` (in-place).
    """
    if alpha is None:
        alpha = 2.0 * rank
    n_blocks = len(dino.blocks)
    for i in block_indices:
        if not (0 <= i < n_blocks):
            raise ValueError(f"block index {i} out of range [0, {n_blocks})")
        blk = dino.blocks[i]
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank=rank, alpha=alpha)
        blk.attn.proj = LoRALinear(blk.attn.proj, rank=rank, alpha=alpha)
    return dino


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
