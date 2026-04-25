# LoRA 微调策略设计

## 背景与动机

当前 Projector 微调策略的瓶颈在于：**DINOv3 主干完全冻结，Projector 只能在固定特征空间里寻找判别方向**。实验结果显示 epoch 0 时正常 patch 与缺陷 patch 的余弦相似度差距仅为 0.030（DINOv3 天然特征），Projector 通过训练把这个差距扩大到了 0.887，但代价是同时略微降低了光照不变性（pos_sim 从 0.995 降到 0.949）。

这说明 Projector 是在**用一个线性映射把两个原本几乎重叠的分布强行推开**，是一种费力的间接方式。LoRA 的思路不同：让主干自身生成判别性更强的特征，再接一个简单投影头完成 L2 归一化。

---

## 核心原理

对 Transformer 中任意线性层 `W ∈ R^(d_out × d_in)`，LoRA 将权重更新分解为：

```
W' = W + ΔW = W + B · A
```

其中 `A ∈ R^(r × d_in)`，`B ∈ R^(d_out × r)`，秩 `r << min(d_in, d_out)`。

初始化：`A ~ N(0, σ²)`，`B = 0`（保证初始 ΔW = 0，不破坏预训练特征）。  
原始权重 `W` 全程冻结，只训练 `A` 和 `B`。

实际前向：

```python
y = W(x) + scaling * B(A(x))     # scaling = alpha / r
```

`alpha` 是一个超参数（通常设为 2r），控制 LoRA 分支的相对幅度。

---

## 目标模块（DINOv3 ViT-B/16 精确路径）

DINOv3 ViT-B/16 的 12 个 block（`dino.blocks[0..11]`）中，每个 block 有四类可注入线性层：

| 路径 | 形状 (out × in) | 作用 |
| ---- | ---- | ---- |
| `blocks[i].attn.qkv` | 2304 × 768 | Q、K、V 合并投影 |
| `blocks[i].attn.proj` | 768 × 768 | 注意力输出投影 |
| `blocks[i].mlp.fc1` | 3072 × 768 | FFN 上投影 |
| `blocks[i].mlp.fc2` | 768 × 3072 | FFN 下投影 |

**推荐注入哪些层：**

LoRA 通常只注入注意力层（`qkv` 和 `proj`），因为注意力的权重决定 patch 之间如何交互，和"对缺陷位置敏感"的目标最直接相关。FFN 层负责逐 patch 的非线性变换，可作为第二优先级。

**推荐注入哪些 block：**

越靠后的 block 越偏向高层语义，越靠前的越偏向低层纹理。缺陷检测需要两者结合，但数据量（66 个物体）有限，建议从最后 4 个 block（`blocks[8..11]`）开始，防止过早破坏通用表征。

---

## 参数量对比

以只注入 `blocks[8..11]` 的 `qkv` + `proj`（注意力层）为例：

| 项 | 计算 |
| ---- | ---- |
| qkv LoRA per block | `768 × r + r × 2304 = 3072r` |
| proj LoRA per block | `768 × r + r × 768 = 1536r` |
| 4 个 block 合计 | `4 × (3072 + 1536) × r = 18432r` |

| rank r | 注意力 LoRA 参数量 | + FFN LoRA 参数量 | 总可训练量 |
| ---- | ---- | ---- | ---- |
| 4 | 74 K | 123 K | 197 K |
| 8 | 147 K | 246 K | 393 K |
| 16 | 295 K | 491 K | 786 K |
| 32 | 590 K | 983 K | 1.57 M |

对比当前 Projector：**2.76 M**（7 层 × ~394 K）

LoRA (r=8, 注意力层) 仅用当前 **5%** 的参数量来微调主干，剩余的冻结权重依然保持 DINOv3 通用表征。

---

## 两个实验方案

### 方案 A：LoRA-only（去掉 Projector）

```
DINOv3 blocks[0..7]  → 完全冻结
DINOv3 blocks[8..11] → LoRA 注入 attn.qkv + attn.proj，rank r=8
                     ↓
             L2-normalize（per patch，无额外线性层）
                     ↓
             余弦距离热力图
```

可训练参数：~147 K  
**目标**：用极少的参数让主干自身生成判别特征，测试 LoRA 是否足以替代 Projector。

### 方案 B：LoRA + 轻量 Projector

```
DINOv3 blocks[8..11] → LoRA（attn.qkv + attn.proj，rank r=8）
                     ↓
        Linear(768→256) → L2-normalize（per layer，无隐层）
                     ↓
             余弦距离热力图
```

可训练参数：~147 K（LoRA）+ ~197 K（Linear × 7 层）= ~344 K  
**目标**：主干通过 LoRA 生成更判别的特征，再通过轻量头降维。

### 与当前方案对比

| 方案 | 可训练参数 | 主干修改 | 投影头 |
| ---- | ---- | ---- | ---- |
| 当前（Projector） | 2.76 M | 无 | MLP 768→384→256 |
| 方案 A（LoRA-only） | ~147 K | attn × 4 blocks | L2-norm |
| 方案 B（LoRA + 轻量头） | ~344 K | attn × 4 blocks | Linear 768→256 |

---

## 实现要点

### LoRA 注入代码骨架

```python
import math
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear          # 冻结
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(x))


def inject_lora(dino, block_indices, rank=8, alpha=16.0, target_attn=True, target_ffn=False):
    """将 LoRA 注入指定 block 的 attn/FFN 线性层。"""
    for i in block_indices:
        blk = dino.blocks[i]
        if target_attn:
            blk.attn.qkv  = LoRALinear(blk.attn.qkv,  rank, alpha)
            blk.attn.proj = LoRALinear(blk.attn.proj, rank, alpha)
        if target_ffn:
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, rank, alpha)
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, rank, alpha)
    return dino
```

### 主干冻结 + LoRA 解冻

```python
# 先全部冻结
for p in dino.parameters():
    p.requires_grad = False

# 注入 LoRA（内部已将 lora_A/lora_B 置为可训练）
inject_lora(dino, block_indices=range(8, 12), rank=8)

# 验证可训练量
trainable = sum(p.numel() for p in dino.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,}")  # 期望约 147 K
```

### 检查点保存

只需保存可训练参数（LoRA 矩阵），无需保存完整主干权重（主干可从原始权重文件恢复）：

```python
# 保存
lora_state = {k: v for k, v in dino.state_dict().items()
              if 'lora_A' in k or 'lora_B' in k}

# 加载（先从原始权重初始化 dino，再 inject_lora，再 load_state_dict）
```

---

## 训练超参数建议

| 项 | 建议值 | 说明 |
| ---- | ---- | ---- |
| LoRA rank | 8 | 先从 r=8 开始；过大有过拟合风险 |
| LoRA alpha | 16 | scaling = 2.0，标准设定 |
| 注入范围 | blocks 8-11 | 先只改后 4 层，避免破坏低层通用特征 |
| 注入目标 | attn.qkv + attn.proj | 先只改注意力；FFN 作为第二步尝试 |
| 学习率 | 1e-4（建议比 Projector 低一个数量级）| LoRA 修改主干，LR 过大会破坏预训练 |
| weight decay | 0.01（比 Projector 低） | 参数少，不需要太强正则 |
| Epochs | 200-500 | 与当前 Projector 策略等量比较 |
| 损失 / 数据 | 不变（pair-margin loss + mydata triplet） | 只换微调模式 |

---

## 预期行为与风险

**预期优势：**
- 主干可以学习"对缺陷纹理敏感"的注意力模式，而不仅仅是在固定特征上找线性方向
- 参数量大幅减少（147 K vs 2.76 M），在小数据集（66 物体）上过拟合风险可能更低
- 初始 ΔW=0 保证了 DINOv3 预训练的特征在 epoch 0 时完全保留

**主要风险：**
- 若 LR 过大，LoRA 会快速破坏主干的光照不变性，导致 pos_sim 大幅下降（比 Projector 策略更严重，因为直接改主干）
- 数据量不足时，LoRA 可能过拟合到 66 个物体的特定缺陷外观，测试时泛化差

**监控指标**（与当前训练相同）：
- `gap = pos_sim − hard_neg_sim`：核心指标，期望从 ~0.03 稳步增长
- `pos_sim`：不应低于 0.90，否则光照鲁棒性破坏过多
- 训练结束后用 `diagnose.py` 对比 LoRA vs Projector vs Baseline 三方热力图

---

## 推荐实施顺序

1. **先跑方案 A（LoRA-only，r=8，blocks 8-11，attn only）** ——参数最少，风险最低，作为 baseline
2. 若方案 A 的 `gap` 和 `pos_sim` 曲线健康，再尝试增大 r（16）或扩展到 blocks 4-11
3. 若方案 A 效果不如 Projector，再尝试方案 B（LoRA + 轻量线性头）
4. 若需要进一步提升，考虑同时注入 FFN 层
