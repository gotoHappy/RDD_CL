# Training Program

## 〇、全局设定

| 项目               | 设定                                                         |
| ------------------ | ------------------------------------------------------------ |
| Backbone           | DINOv3-ViT-B/16（frozen，不更新参数）                      |
| 输入分辨率         | 512×512（DINOv3 原生支持，patch size = 16×16，即patch gird = 32×32 = 1024） |
| Patch 特征维度     | 768                                    |
| Projector 输出维度 | 256                                                          |
| Batch size         | B = 8 个物体                                                 |
dinov3项目已经clone到：/data2/baizeyu/dinov3，权重未下载
------

## 一、数据准备

### 1.1 M2AD 数据预处理

M2AD 提供了同一物体在相同位姿、不同光照下的多张正常图。你需要提前整理出以下索引结构：

```
m2ad_index[object_id] = {
    "pose_groups": {
        pose_id_0: {
            "light_A": "path/to/img_0_lightA.png",
            "light_B": "path/to/img_0_lightB.png",
            "light_C": "path/to/img_0_lightC.png",
            ...
        },
        pose_id_1: { ... },
        ...
    },
    "foreground_masks": {
        pose_id_0: "path/to/mask_0.png",
        ...
    }
}
```

**步骤**：

1. 遍历 M2AD 所有物体或json文件，按物体-位姿-光照三级索引建表
2. 前景 mask 用 BiRefNet 批量生成，存在M2AD_mask中
3. 所有图像 resize 到 518×518（DINOv2 输入尺寸），前景 mask 同步 resize 到 37×37（patch grid 尺寸），前景面积占比 ≥ 50% 的 patch 标记为前景 patch  ？



### 

------

## 二、在线伪异常生成（针对 M2AD）

M2AD 没有异常图，需要 online 合成。每次从 dataloader 取出 N1 时，同步生成伪异常图 A1_synth 和伪缺陷 mask M_synth。

### 2.1 生成流程

对一张正常图 N1（518×518）：

**步骤 1：生成不规则区域 mask**

- 用 Perlin 噪声生成一张 518×518 的连续噪声图
- 对噪声图做二值化（阈值 0.5），得到不规则形状的二值 mask M_raw
- 控制 mask 面积占前景面积的 5%-30%（若超出则调整 Perlin 噪声的阈值）
- 对 M_raw 做 5×5 高斯模糊得到软边缘 mask M_soft（用于融合时平滑过渡）

**步骤 2：在 mask 区域内施加异常**

随机选一种（均匀概率）：

| 方法     | 操作                                                         | 效果模拟             |
| -------- | ------------------------------------------------------------ | -------------------- |
| 纹理替换 | 从 DTD 数据集随机取一张纹理图，crop 到 mask 区域大小，用 M_soft 做 alpha blending 融合到 N1 | 锈斑、污渍、涂层脱落 |
| CutPaste | 从同 batch 其他物体的图上随机 crop 一个区域，贴到 mask 位置  | 外来异物、贴纸       |
| 颜色扰动 | 在 mask 内对像素做 HSV 空间的随机偏移（H±30°, S×[0.5,1.5], V×[0.5,1.5]） | 色斑、褪色           |
| 噪声叠加 | 在 mask 内加高斯噪声（σ=30-80）或椒盐噪声                    | 粗糙表面、磨损       |
| 模糊     | 在 mask 内做高斯模糊（kernel 11-21）                         | 磨损、抛光痕迹       |

DTD数据集已下载: /data2/baizeyu/dtd

**步骤 3：生成 patch 级别的缺陷标注**

将 M_raw（518×518）下采样到 37×37 的 patch grid。每个 patch 位置计算 M_raw 在该 patch 区域内的面积占比：

- defect ratio > 0.3 → 标记为 `defect`
- defect ratio < 0.05 → 标记为 `clean`
- 其余 → 标记为 `ignore`

### 2.2 关键参数

| 参数                 | 值              | 说明                   |
| -------------------- | --------------- | ---------------------- |
| Perlin 噪声 octaves  | 4-6             | 控制 mask 边缘复杂度   |
| Perlin 噪声 scale    | 2-6（随机）     | 控制异常区域大小       |
| 异常面积占前景面积比 | 5%-30%          | 太小学不到，太大不真实 |
| Alpha blending 强度  | 0.4-0.8（随机） | 控制异常明显程度       |

### 2.3 注意事项

- 伪异常只生成在前景区域内，背景区域不施加任何变化
- 每个 epoch 对同一张 N1 随机生成不同的伪异常（Perlin 种子随机），相当于无限增强
- A1_synth 的光照条件与 N1 完全相同（因为是在 N1 上直接操作的），不影响光照不变性学习

------

## 三、Dataloader 采样策略

### 3.1 每个 batch 的构成

每个 batch 采 B=8 个物体，每个物体产出一个三元组 (N1, N2, A1)。

**物体来源分配**：

| 来源     | 每 batch 数量 | 三元组构成                                                   |
| -------- | ------------- | ------------------------------------------------------------ |
| M2AD     | 8             | N1, N2=同位姿不同光照正常图, A1=N1 上 online 合成的伪异常    |



### 3.2 采样一个 M2AD 三元组的流程

```
1. 随机选一个物体 obj
2. 随机选该物体的一个位姿 pose（要求该位姿有 ≥2 种光照）
3. 从该位姿的可用光照中随机选两个不同光照 light_a, light_b
4. N1 = img[obj][pose][light_a]
5. N2 = img[obj][pose][light_b]
6. A1_synth = synthesize_anomaly(N1)  # online 生成
7. 返回 (N1, N2, A1_synth, foreground_mask, defect_mask_synth)
```


## 四、对比关系与损失函数

### 4.1 特征提取

一个 batch 的 24 张图（8 物体 × 3 张）全部过 DINOv2 + Projector：

```
输入: 24 张 518×518 图像
     ↓
DINOv2 frozen 前向 → 24 × L × 37 × 37 × 768
     ↓
取指定的c层特征 (用户通过配置文件选择)
     ↓
Projector (768 → 384 → 256, 中间 GELU, 最后 L2 normalize)
     ↓
输出: 24 × c × 37 × 37 × 256 (单位超球面上)
```

### 4.2 前景 patch 筛选

对每张图，只保留前景 mask 标记为前景的 patch。设第 b 个物体的 N1 有 K_b 个前景 patch。

### 4.3 构建对比对

以第 b 个物体的 N1 的第 i 个前景 patch 特征 z_i 为 anchor：

**正例集**：

| 编号 | 来源               | 条件           | 描述                   |
| ---- | ------------------ | -------------- | ---------------------- |
| 正例 | N2 的第 i 个 patch | 该位置也是前景 | 同位置、不同光照、正常 |

p2 不一定存在（该位置可能是 defect 或 ignore），P(i) 的大小为 1 或 2。

**负例集 **：

| 编号          | 来源                              | 数量            | 描述                   |
| ------------- | --------------------------------- | --------------- | ---------------------- |
| 硬负例        | A1 的第 i 个 patch                | 0 或 1          | 该位置是 defect 时存在 |
| in-batch 负例 | 其余 7 个物体的全部 N1 前景 patch | Σ_{b'≠b} K_{b'} | 跨物体 patch           |

**不作为负例的**：

- 同物体其他位置的 patch（避免同质纹理被错误推远）
- 背景 patch（不参与任何关系）
- ignore 区域的 patch（边界模糊，信号不清晰）

### 4.4 损失函数设计

#### 符号定义

一个 batch 有 B=8 个物体，编号 b ∈ {1,...,8}。每个物体有三张图 (N1_b, N2_b, A1_b)，全部过 DINOv2 + Projector 后得到 L2 归一化的 patch 特征。

对物体 b，设其前景 patch 索引集合为 F_b（大小为 K_b）。对 patch 位置 i ∈ F_b：

| 符号      | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| z_i^b     | N1_b 的第 i 个 patch 特征（anchor）                          |
| z_i^{b+}  | N2_b 的第 i 个 patch 特征（正例）                            |
| z_i^{b,a} | A1_b 的第 i 个 patch 特征                                    |
| d_i^b     | A1_b 的第 i 个 patch 的缺陷标签：`defect` / `clean` / `ignore` |

#### 每个 anchor 的对比集

对 anchor z_i^b，它参与的所有关系如下：

**正例**（恰好 1 个）：z_i^{b+}，即 N2 同位置 patch。

**负例**（分两类）：

第一类——硬负例。当且仅当 d_i^b = defect 时存在，为 z_i^{b,a}。数量为 0 或 1。

第二类——in-batch 负例。所有其他物体的 N1 前景 patch 特征：

```
N_inbatch(b) = { z_j^{b'} | b' ≠ b, j ∈ F_{b'} }
```

大小约为 Σ_{b'≠b} K_{b'}，通常在 3000-4000 之间。

#### 三种 anchor 情况的损失

根据 d_i^b 的取值，每个 anchor 落入三种情况之一：

**情况 1：d_i^b = clean 或位置 i 在 A1 上不是前景**

该位置没有硬负例，负例集只有 in-batch 负例。

```
L_i^b = -log [ exp(s_pos / τ) / ( exp(s_pos / τ) + Σ_{n ∈ N_inbatch(b)} exp(s_n / τ) ) ]
```

其中 s_pos = z_i^b · z_i^{b+}，s_n = z_i^b · z_n。

这种 anchor 只提供光照不变性的学习信号。

**情况 2：d_i^b = defect**

该位置有一个硬负例 z_i^{b,a}，负例集 = 硬负例 + in-batch 负例。

```
L_i^b = -log [ exp(s_pos / τ) / ( exp(s_pos / τ) + exp(s_hard / τ) + Σ_{n ∈ N_inbatch(b)} exp(s_n / τ) ) ]
```

其中 s_hard = z_i^b · z_i^{b,a}。

这种 anchor 同时提供光照不变性和缺陷敏感性的学习信号。s_hard 通常较大（缺陷区域和正常区域在外观上仍有相似性），所以 exp(s_hard / τ) 在分母中占比大，对梯度的贡献也大——这正是"硬"的含义。

**情况 3：d_i^b = ignore**

该 anchor 不参与任何损失计算，直接跳过。

#### 统一公式

把三种情况合并为一个公式。定义指示函数 h_i^b：当 d_i^b = defect 时 h_i^b = 1，否则 h_i^b = 0。

```
L_i^b = -log [ exp(s_pos / τ) / D_i^b ]

D_i^b = exp(s_pos / τ) + h_i^b · exp(s_hard / τ) + Σ_{n ∈ N_inbatch(b)} exp(s_n / τ)
```

当 h_i^b = 0 时，硬负例项自然消失，退化为情况 1。

#### Batch 总损失

对所有参与计算的 anchor（排除 ignore）取平均：

```
L = (1 / M) × Σ_b Σ_{i ∈ F_b, d_i^b ≠ ignore} L_i^b
```

其中 M 是参与计算的 anchor 总数。

#### 实际计算流程（矩阵化）

逐 patch 算 InfoNCE 太慢，实际要矩阵化。以下是一个 batch 内的计算步骤：

**步骤 1：收集所有 anchor 特征**

把 8 个物体的 N1 前景 patch 拼成一个矩阵：

```
Z_anchor: (M, 256)     # M ≈ 4000，所有物体 N1 前景 patch
```

同时记录每个 anchor 属于哪个物体：ownership 数组，长度 M。

**步骤 2：收集所有正例特征**

对每个 anchor z_i^b，它的正例是 N2_b 同位置 patch。由于 anchor 和正例是一一对应的，正例矩阵与 anchor 矩阵行对齐：

```
Z_pos: (M, 256)        # 与 Z_anchor 逐行对应
```

**步骤 3：计算正例相似度**

```
S_pos: (M,)
S_pos[k] = Z_anchor[k] · Z_pos[k]    # 逐行点积
```

**步骤 4：计算 in-batch 负例相似度**

anchor 与所有 anchor 的相似度矩阵：

```
S_all: (M, M)
S_all = Z_anchor @ Z_anchor.T         # 全局点积矩阵
```

然后用 ownership 数组构造 mask，屏蔽同物体的 patch：

```
mask_same_obj: (M, M)  布尔矩阵
mask_same_obj[k1, k2] = (ownership[k1] == ownership[k2])

S_all[mask_same_obj] = -inf           # 同物体 patch 不作为负例
S_all 对角线 = -inf                    # 自己不作为负例
```

**步骤 5：加入硬负例**

收集硬负例特征。对每个 anchor，如果它有硬负例，计算相似度：

```
Z_hard: (M, 256)       # 没有硬负例的行填零
H_mask: (M,)           # 布尔，标记哪些 anchor 有硬负例

S_hard: (M,)
S_hard[k] = Z_anchor[k] · Z_hard[k]   # 逐行点积
```

**步骤 6：计算每个 anchor 的损失**

```
# 分子
numerator = exp(S_pos / τ)             # (M,)

# 分母：正例 + 硬负例 + in-batch 负例
denom_inbatch = Σ_j exp(S_all[k, j] / τ)   # 对每行求和，(M,)
denom_hard = H_mask * exp(S_hard / τ)       # 没有硬负例的行为 0，(M,)
denominator = numerator + denom_hard + denom_inbatch    # (M,)

# 每个 anchor 的损失
losses = -log(numerator / denominator)   # (M,)
```

**步骤 7：排除 ignore anchor，取平均**

```
valid_mask: (M,)       # 布尔，d_i^b ≠ ignore 的 anchor

L = losses[valid_mask].mean()
```



### 4.5 温度参数 τ

| 策略                   | 值                                     | 说明                            |
| ---------------------- | -------------------------------------- | ------------------------------- |
| 固定温度（推荐起步）   | τ = 0.07                               | DINO/MoCo 的常用值              |
| 可学习温度（可选尝试） | 初始化 τ = 0.07，log(τ) 作为可训练参数 | CLIP 用的策略，让模型自适应调整 |

建议先用固定 0.07 跑通，如果发现训练不稳定（loss 震荡）再调大到 0.1。

### 4.6 硬负例的权重处理

硬负例数量远少于 in-batch 负例（一张图上可能只有 10-30 个 defect patch，而 in-batch 负例有 ~700 个）。在 InfoNCE 中硬负例天然会因为与 anchor 相似度高而获得更大的梯度，无需额外加权。

但如果你发现缺陷敏感性学得不够，可以在负例的 exp 项前加一个权重：硬负例权重 w_hard = 3-5，in-batch 负例权重 w_inbatch = 1。修改后的分母变为：

```
Σ_{p} exp(sim/τ)  +  w_hard × Σ_{n_hard} exp(sim/τ)  +  w_inbatch × Σ_{n_inbatch} exp(sim/τ)
```

这是可选优化，不是必须的。

------

## 五、Projector 架构

```
Projector:
  Linear(768, 384)
  GELU
  Linear(384, 256)
  L2Normalize
```

- 不加 BatchNorm（patch 级特征的 BN 在小 batch 下统计量不稳定）
- 最后一层不加激活函数，直接 L2 normalize 到单位超球面
- 参数量：768×384 + 384×256 ≈ 400K，非常轻量

------

## 六、训练超参数

| 参数                 | 值                                    | 说明                                     |
| -------------------- | ------------------------------------- | ---------------------------------------- |
| 优化器               | AdamW                                 |                                          |
| 学习率               | 1e-3                                  | 仅训练 projector，可以用较大学习率       |
| 学习率调度           | 线性 warmup 5 epoch + cosine decay    |                                          |
| 权重衰减             | 0.05                                  |                                          |
| Epochs               | 100                                   | 视 M2AD 数据量调整，监控验证指标决定早停 |
| Batch size (物体数)  | 8                                     |                                          |
| 每物体图像数         | 3 (N1, N2, A1)                        |                                          |
| 每 batch 总图像      | 24                                    |                                          |
| 每 batch 总 patch 数 | ~24 × 500 ≈ 12000（前景 patch）       |                                          |
| 每 batch anchor 数   | ~8 × 500 = 4000（仅 N1 的前景 patch） |                                          |
| 每 anchor 正例数     | 1-2                                   |                                          |
| 每 anchor 负例数     | ~3500 (in-batch) + 0-1 (硬负例)       |                                          |
| 温度 τ               | 0.07                                  |                                          |

------

## 七、数据增强

### 7.1 对 N1, N2 的增强（轻度，不破坏位姿对应关系）

对 N1 和 N2 做**完全相同的空间变换**（保持 patch 对应关系），但允许**独立的**颜色变换：

| 增强类型 | 操作                             | N1 和 N2 是否同步 |
| -------- | -------------------------------- | ----------------- |
| 空间     | 随机水平翻转                     | 同步              |
| 空间     | 随机 resize crop (scale 0.8-1.0) | 同步              |
| 颜色     | 亮度 ×[0.8, 1.2]                 | 各自独立          |
| 颜色     | 对比度 ×[0.8, 1.2]               | 各自独立          |
| 颜色     | JPEG 压缩 quality [70, 100]      | 各自独立          |

颜色增强各自独立的原因：进一步增大 N1 和 N2 的光照差异，让模型学到更强的光照不变性。

### 7.2 对 A1 的增强

A1（无论是合成的还是真实的）做与 N1 **完全相同的空间变换**（保持 patch 对应关系）。颜色增强与 N1 同步（因为 A1 应该与 N1 同光照）。

### 7.3 前景 mask 和 defect mask 的同步

所有空间变换必须同步应用到前景 mask 和 defect mask 上。

------

## 八、验证与监控

### 8.1 训练过程中监控的指标

| 指标                        | 计算方式                           | 健康范围           | 异常信号                          |
| --------------------------- | ---------------------------------- | ------------------ | --------------------------------- |
| Training loss               | 每 100 步平均                      | 单调下降后趋于平稳 | 不下降或震荡                      |
| 正例平均余弦相似度          | 所有 anchor 与其正例的 sim 均值    | 0.7-0.95           | < 0.5 说明没学到，> 0.99 可能坍缩 |
| 硬负例平均余弦相似度        | 所有 anchor 与其硬负例的 sim 均值  | 0.1-0.5            | > 0.7 说明缺陷分不开              |
| in-batch 负例平均余弦相似度 | 随机采样的跨物体 patch 对 sim 均值 | 0.0-0.3            | > 0.5 说明特征没有判别性          |
| 特征维度标准差              | 所有 patch 特征在每个维度的 std    | 每个维度 > 0.01    | 若多个维度 std → 0 则特征坍缩     |



------

## 九、训练完成后的产出

| 产出          | 内容                                                         |
| ------------- | ------------------------------------------------------------ |
| 模型权重      | Projector 的参数（~400K 参数，< 2MB）                        |
| 推理 pipeline | DINOv2 frozen 前向 → 取 layer 11 → Projector → L2 normalize → 256维 patch 特征 |
| 使用方式      | 对 ref 和 query 各提取 patch 特征，计算逐 patch 余弦距离，上采样为异常热力图 |

