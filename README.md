# Robust Defect Detection — Contrastive Learning

## 项目目标

给定同一物体的**正常参考图**（ref）和**待检查查询图**（query），输出像素级缺陷热力图和二值缺陷 mask。

```
输入：ref.png  +  query.png
输出：heatmap.png  /  pred_mask.png  /  blend.png
```

适用场景：工业质检，每个物体有一张理想光照下的 ref，query 可能来自不同光照，可能含有真实缺陷。

---

## 核心思路

**patch 级余弦距离检测**

用 DINOv3 ViT-B/16 提取 ref 和 query 的 patch 特征，经过一个轻量 projector 映射到判别子空间，然后对应位置的余弦距离即为该 patch 的异常分数。

```
ref  → DINOv3 → Projector → {z_ref_i}
query → DINOv3 → Projector → {z_qry_i}

anomaly_score_i = 1 − cos(z_ref_i, z_qry_i)
```

正常区域：ref 和 query 特征方向接近，分数低。  
缺陷区域：特征方向偏离，分数高。

**为什么不直接用 DINO 原始特征？**  
DINOv3 原始特征的正常/缺陷分数差异非常小（实测 Δcos ≈ 0.01）。Projector 通过有监督训练主动扩大这个差距，使得缺陷 patch 在投影空间里显著偏离正常方向。

---

## 模型结构

```
输入图像: 512×512  →  patch grid: 32×32 = 1024 patches
                         │
                  DINOv3 ViT-B/16 (冻结)
                  取第 5-11 层共 7 个 block 输出
                  每层特征维度: 768
                         │
                  PatchProjector × 7 (可训练, ~394K params/层, 合计 ~2.76M)
                  Linear(768→384) → GELU → Linear(384→256) → L2-normalize
                         │
                 7 × (B, 256, 32, 32)  逐层余弦距离 → 加权融合 → 上采样 → 热力图
```

只训练 Projector，DINOv3 主干全程冻结。

---

## 训练策略

### 数据

训练数据来自 `mydata/`（120 个物体，其中 66 个包含真实缺陷 query）。

每个物体的文件结构：
```
mydata/obj_XXXX/
    ref.png                                   # 理想光照正常参考图
    ref_fg_mask.png                           # 前景二值 mask
    queries/
        obj_XXXX__normal__none__light_N.png   # 不同光照正常 query
        obj_XXXX__anomaly__<type>__light_N.png         # 真实缺陷 query
        obj_XXXX__anomaly__<type>__light_N_defect_mask.png  # GT 缺陷 mask
```

每个训练样本（三元组）：
```
N1 = ref.png          （理想光照正常参考）
N2 = random normal query （不同光照正常，训练跨光照不变性）
A1 = random anomaly query（真实缺陷，带 GT mask）
```

### 损失函数（Pair-Margin Loss）

不使用 InfoNCE，不使用跨物体负样本。在前景 patch 上直接施加两类约束：

```
对于 clean anchor（A1 对应位置无缺陷）:
    L_clean = max(0,  β − cos(z_N1, z_N2))        β = 0.95
    → 正常图不同光照特征拉近

对于 defect anchor（A1 对应位置有缺陷）:
    L_defect = max(0,  α − (cos(z_N1, z_N2) − cos(z_N1, z_A1)))   α = 0.30
    → 要求正常对的相似度比缺陷对高出至少 α

总损失 = (1.0 × mean(L_clean) + 3.0 × mean(L_defect)) / 4.0
```

**设计意图**：`L_defect` 直接优化"正常和缺陷的余弦距离之差"，和推理时使用的指标完全对齐，没有 InfoNCE 的温度饱和问题。

### 训练超参数

| 项 | 值 |
| ---- | ---- |
| Backbone | DINOv3 ViT-B/16（完全冻结） |
| Projector 层数 | 7 层（block 5-11），共 2.76M 可训练参数 |
| 输入尺寸 | 512×512 |
| Batch size | 8 物体 |
| 优化器 | AdamW, lr=1e-3, wd=0.05 |
| 调度器 | Linear warmup 5 ep + cosine → 1% |
| 总 epochs | 500 |

---

## 数据路径

```
/data2/baizeyu/RDD_CL/
    mydata/                   # 训练集（120 物体）
    mytestdata/               # 测试集（5 物体，暂无 GT mask）
    outputs/                  # 训练和推理输出

/data2/baizeyu/dinov3/
    dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth   # DINOv3 预训练权重
```

---

## 运行命令

### 环境

```bash
conda activate RDD
cd /data2/baizeyu/RDD_CL
```

### 训练

```bash
python scripts/train_mydata.py scripts/configs/train_mydata.yml
```

输出保存到 `outputs/dino3_vitb16_mydata_margin_500/`：
- `best.pth` / `last.pth` — 最优和最终 checkpoint
- `checkpoints/{epoch}.layer.pth` — 按 `save-checkpoint-freq` 频率保存
- `logs/{epoch}.layer.pkl` — 每 epoch 的 loss + metrics
- `args.json` — 完整训练配置快照

训练过程 tqdm 实时展示：`pos`（正样本相似度）、`hn`（硬负例相似度）、`gap`（关键指标，目标从 0.01 增长到接近 0.30）。

### 推理（使用训练好的 Projector）

```bash
python scripts/infer_contrastive.py outputs/dino3_vitb16_mydata_margin_500/best.pth \
    --dataset-root mytestdata \
    --output outputs/infer_contrastive \
    --threshold 0.5 \
    --device cuda
```

每个样本输出目录包含：`ref.png` / `query.png` / `heatmap.png` / `score_map.png` / `pred_mask.png` / `blend.png`。

### 推理（DINO 原始特征 Baseline，不使用 Projector）

```bash
python scripts/infer_baseline.py outputs/dino3_vitb16_mydata_margin_500/best.pth \
    --dataset-root mytestdata \
    --output outputs/infer_baseline \
    --device cuda
```

### 可视化（在 mydata 训练集上对比 GT）

```bash
python scripts/visualize_contrastive.py outputs/dino3_vitb16_mydata_margin_500/best.pth \
    --dataset mydata \
    --output outputs/vis_mydata
```

展示：Reference / Query / Heatmap / Blend / Pred Mask / GT Mask（mydata 有标注时）。

### 定量评估

```bash
python scripts/evaluate_contrastive.py outputs/dino3_vitb16_mydata_margin_500/best.pth \
    --threshold 0.5 \
    --output outputs/eval.json
```

### 诊断对比（Contrastive vs Baseline 差异分析）

```bash
python scripts/diagnose_m2ad.py outputs/dino3_vitb16_mydata_margin_500/best.pth \
    --dataset-root mytestdata \
    --output outputs/diagnose
```

对每个测试样本生成 `compare.png`（共享色标的并排热力图 + Δ图 + 分数分布直方图）和训练曲线图 `training_curves.png`。如有 GT mask 可传 `--gt-mask-root <path>` 计算 AUROC。

---

## 配置说明（`scripts/configs/train_mydata.yml`）

| 段 | 关键字段 | 说明 |
| ---- | ---- | ---- |
| `model` | `layers` | 提取哪些 DINOv3 block，默认 `[5,6,7,8,9,10,11]` |
| | `dino-weights-path` | DINOv3 本地权重路径 |
| `dataset` | `root` | mydata 根目录 |
| | `batch-size` | 每 batch 物体数，默认 8 |
| `margin-loss` | `margin-triplet` | α，缺陷 anchor 的 gap 目标，默认 0.30 |
| | `margin-positive` | β，clean anchor 的 pos_sim 目标，默认 0.95 |
| | `defect-weight` | 缺陷损失权重，默认 3.0（缺陷 patch 数量少，需要上调权重） |
| | `patch-defect-thresh` | GT mask 池化后 > 此值的 patch 标记为 defect，默认 0.30 |
| `optimizer` | `epochs` | 默认 500 |
| `inference` | `layer-weights` | 推理端多层融合权重 |
| `wandb` | `mode: disabled` | 关闭在线记录；改为 `online` 开启 |

---

## 训练健康指标

| 指标 | 含义 | 健康状态 |
| ---- | ---- | ---- |
| `gap` | defect anchor 上 pos_sim − hard_neg_sim | 从 ~0.01 稳定增长，目标接近 α=0.30 |
| `pos_sim` | clean anchor 上正常图对的余弦相似度 | 维持 ≥ 0.95，下跌说明 projector 破坏了正常对齐 |
| `hard_neg_sim` | defect anchor 上 N1 vs A1 余弦相似度 | 随训练下降，最终应低于 pos_sim − α |
| `loss_defect` | 缺陷损失 | 单调下降 |
| `loss_clean` | 正样本损失 | 初期为 0（DINO 特征天然 pos_sim 很高），正常 |
| `n_defect` | 每 batch 缺陷 anchor 数量 | 应在数十到数百级别，过小说明 `patch-defect-thresh` 设置过严 |

---

## 项目文件结构

```text
RDD_CL/
├── mydata/                         # 训练集（120 物体）
├── mytestdata/                     # 测试集（5 物体）
├── src/robust_defect_detection/
│   ├── datasets/
│   │   ├── mydata_triplet.py       # ★ 核心：mydata 三元组数据集
│   │   ├── m2ad.py                 # M2AD 数据集（实验备用）
│   │   └── contrastive_triplet.py  # 旧 mydata 数据集（兼容旧 checkpoint）
│   └── models/
│       ├── backbone_dinov3.py      # DINOv3 本地加载器
│       ├── backbone_dinov2.py      # DINOv2 加载器（兼容旧 checkpoint）
│       └── contrastive.py          # MultiLayerContrastiveModel + PatchProjector
├── scripts/
│   ├── train_mydata.py             # ★ 核心训练脚本
│   ├── infer_contrastive.py        # 推理（使用 Projector）
│   ├── infer_baseline.py           # 推理（仅 DINO 原始特征）
│   ├── visualize_contrastive.py    # 可视化
│   ├── evaluate_contrastive.py     # 定量评估
│   ├── diagnose_m2ad.py            # Contrastive vs Baseline 诊断对比
│   └── configs/
│       ├── train_mydata.yml        # ★ 核心配置
│       ├── train_m2ad_pair_mydata.yml   # 实验：M2AD pair 正则 + mydata
│       └── train_m2ad_mydata.yml        # 实验：M2AD 合成 + mydata 混合
└── outputs/
    └── dino3_vitb16_mydata_margin_500/  # 默认训练输出目录
```

标 ★ 的是当前最优 pipeline 涉及的核心文件。
