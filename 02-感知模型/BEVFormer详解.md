---
tags: [BEV, transformer, model]
created: "2026-07-21"
updated: "2026-08-21"
---

# BEVFormer 详解

> **论文**: BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers (ECCV 2022)
> **机构**: 上海 AI Lab / 上海交大 / 商汤
> **代码**: [fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | BEVFormer 的完整机制：BEV Query、时空注意力（Temporal Self-Attention + Spatial Cross-Attention）、Deformable Attention 数学、代码走读、训练细节、消融实验、面试追问 |
| **需要的前置知识** | [[Transformer架构详解]]（Attention/Cross-Attention/Deformable）、[[3D视觉与投影几何]]（投影数学）、[[BEV感知全景]]（BEV 认知）、[[BEVDet与BEVDepth]]（LSS 对照） |
| **学完之后你能** | ① 画出 BEVFormer 完整数据流；② 解释"为什么用 Query 而不是直接投影"；③ 看懂 Deformable Attention 公式；④ 回答"相机故障会怎样""为什么只用 yaw 对齐"等深度追问 |
| **预计阅读时间** | 120-150 分钟（本篇密度高，建议分 2-3 次读） |
| **难度** | ⭐⭐⭐（基础理论之后的第一篇硬核模型笔记） |

> [!tip] 阅读建议
> **第一遍**只读第一节"大白话总览"和第二节架构图，建立整体印象；**第二遍**再啃公式和代码；**第三遍**回来做自测题和练习。直接硬啃公式容易劝退。

---

## 〇、大白话总览（入门速览，先读这个！）

### BEVFormer 在做什么？

输入：6 个相机的当前帧图像 + 上一帧的 BEV 特征。
输出：一张 200×200 的 BEV 特征图（每个格子代表地面上 0.512m × 0.512m 的区域）。
用途：在这张 BEV 特征上接检测头（找车/人）、分割头（车道线）、运动头（预测运动）。

### 核心思想（一句话）

**BEV 网格上每个格子是一个"可学习的问题（Query）"，它通过注意力机制主动去图像里"问"自己需要的信息，同时参考上一帧的 BEV 特征记住"过去"。**

### 三个关键问题的直觉答案

1. **为什么要 Query？不能直接把图像投影到 BEV 吗？**
   > 因为**每个像素的深度未知**（单目相机），无法精确投影——你只知道像素对应"一条射线"，不知道它到底在射线上哪个位置。Query 机制的思路是：**我不猜深度，我让网络自己学"该去图像的哪个位置看"**。

2. **为什么分"空间"和"时序"两个注意力？**
   > 一个 BEV 格子需要两类信息：**现在**（当前图像上这个位置有什么）和**过去**（上一帧这个位置有什么）。空间注意力管"现在"，时序注意力管"过去"，各司其职。

3. **为什么用 Deformable（可变形）注意力？**
   > BEV 有 40000 个格子，图像特征有几十万个位置，全连接注意力是 O(N²) 直接算不动。**可变形注意力让每个格子只去图像上"重点的 4 个点"采样**，复杂度降了几个数量级（见第三节公式）。

### 和 LSS 范式的本质差异（对照表）

| | LSS (BEVDet) | BEVFormer |
|---|---|---|
| **深度处理** | 显式预测深度分布 → 投影 | Attention 隐式学习采样位置 |
| **核心算子** | Voxel Pooling (sum pooling) | Deformable Cross-Attention |
| **深度监督** | BEVDepth 需要显式深度真值 | 不需要（端到端学习） |
| **时序机制** | Concat 历史 BEV (BEVDet4D) | Temporal Self-Attention |
| **对相机标定的依赖** | 强依赖 (外参用于 3D→2D 投影) | 中依赖 (参考点投影) |
| **显存** | 视锥体特征大 | BEV Query 小 ($200^2 \times 256$) |

> [!note] 记住这张表的核心
> **LSS 是"先量距离再搬特征"（显式、可解释、怕深度错）；BEVFormer 是"问图像要答案"（隐式、端到端、怕算力贵）。** 两条路线的所有差异都源于"深度怎么处理"这一个分叉点。

---

## 一、核心思想（展开版）

一句话：**用可学习的 BEV Queries，通过可变形交叉注意力 (Deformable Cross-Attention) 去多相机图像特征上"询问"每个位置的视觉信息，并结合历史 BEV 特征做时序融合。**

> [!question] 追问自己：为什么要 Query？不能直接把图像特征投影到 BEV 吗？
> 因为每个像素的深度未知（单目相机），无法精确投影。Query 机制通过 Attention 自主学习"去图像的哪个位置看"来解决这个问题。**这就是"隐式投影"和"显式投影"的分水岭**（[[3D视觉与投影几何]] 的第三节）。

---

## 二、完整架构

### 2.1 架构参数速查表

| 参数 | 值 | 含义 |
|------|-----|------|
| $H_{bev}, W_{bev}$ | $200 \times 200$ | BEV 网格分辨率 |
| $d_{model}$ | $256$ | 隐层维度 |
| $n_{head}$ | $8$ | 多头注意力头数 |
| $n_{enc}$ | $6$ | Encoder Layer 数量 |
| $n_{level}$ | $4$ | 多尺度特征层级（FPN输出） |
| $n_{points}$ | $4$ | 每个 Query 的 Deformable 采样点数 |
| $n_{height}$ | $4$ | Z 轴采样高度 (-5m 到 3m) |
| $d_{ff}$ | $512$ | FFN 隐层维度 |
| BEV 范围 | $[-51.2m, 51.2m]^2$ | 物理范围 |
| grid 分辨率 | $0.512m$ | 每个 BEV cell 的物理尺寸 |

> [!note] 一组数字建立规模感
> BEV 有 200×200 = **40000 个 query**，每个 query 要采样 4 个高度 × 4 个采样点，投影到 6 个相机、4 个尺度特征——所以一次前向要处理约 40000 × 4 × 4 × 6 × 4 ≈ **1500 万次采样**。这就是 BEVFormer 慢（2.5 FPS）的原因，也是 Deformable Attention 存在的意义。

### 2.2 整体 Pipeline

```
            t-1 时刻              t 时刻 (当前)
               │                      │
        ┌──────▼──────┐         ┌─────▼──────┐
        │ 历史 BEV 特征 │        │ 6 相机图像  │
        │ B_{t-1}      │        │ {I_1..I_6} │
        │ [200,200,256]│        └─────┬──────┘
        └──────┬──────┘               │
               │                 ResNet/Swin + FPN
               │                 ┌─────▼──────┐
               │                 │ 多尺度特征  │
               │                 │ F_l ∈ R^{H_l×W_l×256} │
               │                 │ l=1,2,3,4   │
               │                 └─────┬──────┘
               │                       │
        ┌──────▼───────────────────────▼──────┐
        │     BEVFormer Encoder (×6 layers)   │
        │                                      │
        │  ┌───────────────────────────┐      │
        │  │ Temporal Self-Attention   │      │
        │  │ Q=B_cur, K,V=warp(B_prev)│      │
        │  └───────────┬───────────────┘      │
        │              ↓                      │
        │  ┌───────────────────────────┐      │
        │  │ Spatial Cross-Attention   │      │
        │  │ Q=BEV_query, K,V=img_feat │      │
        │  │ 4 reference_points →      │      │
        │  │ 投影到各相机 → 采样 K=4   │      │
        │  └───────────┬───────────────┘      │
        │              ↓                      │
        │  ┌───────────────────────────┐      │
        │  │ FFN (MLP + GELU)          │      │
        │  └───────────────────────────┘      │
        └──────────────────┬───────────────────┘
                           │
                    ┌──────▼──────┐
                    │ BEV Features │
                    │ [200,200,256]│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Detection    Map Seg      Motion
         Head         Head         Head
```

> [!note] 每个 Encoder Layer 里发生了什么（逐层推进）
> **① Temporal Self-Attention（先看过去）** → **② Spatial Cross-Attention（再看现在）** → **③ FFN（自己消化）**。
> 6 层重复这个"过去 → 现在 → 消化"的循环，每层都让 BEV 特征更丰富。**先时序后空间**的顺序是有讲究的：先和历史特征对齐融合（获得"延续性"），再去图像里补充"当前证据"。

---

## 三、公式推导

### 3.1 Deformable Attention 完整数学

标准 Multi-Head Attention 有 $O(N^2)$ 的问题（$N=40000$ 个 BEV query）。Deformable Attention 只采样 $K$ 个点：

$$\text{DeformAttn}(z_q, p_q, \{x^l\}_{l=1}^L) = \sum_{m=1}^{M} W_m \left[ \sum_{l=1}^{L} \sum_{k=1}^{K} A_{mlqk} \cdot W_m' x^l(p_q + \Delta p_{mlqk}) \right]$$

符号定义：
- $z_q \in \mathbb{R}^{d}$: 第 $q$ 个 BEV Query（$q = 1..40000$）
- $p_q = (x_q, y_q)$: BEV Query 在 BEV 平面的参考位置
- $x^l \in \mathbb{R}^{C \times H_l \times W_l}$: 第 $l$ 级特征图（$l=1..4$）
- $L=4$: 多尺度特征层级数
- $K=4$: 每层每头的采样点数
- $M=8$: 注意力头数
- $\Delta p_{mlqk} \in \mathbb{R}^2$: 可学习的采样偏移（通过 Linear 层从 $z_q$ 预测）
- $A_{mlqk} \in [0,1]$: 注意力权重（softmax over ${l,k}$）

**复杂度**: $O(N \cdot L \cdot K \cdot C)$ = $O(40000 \cdot 4 \cdot 4 \cdot 256)$ = $O(163.8\text{M})$，远小于标准 Attention 的 $O(40000^2 \cdot 256) = O(409.6\text{B})$——**差 2500 倍**。

> [!note] 公式的中文翻译（别被符号吓到）
> "每个 BEV query $z_q$，以它的位置 $p_q$ 为中心，在每层特征 $l$ 上采样 $K=4$ 个点（位置 = 参考点 + 学出来的偏移 $\Delta p$），用学出来的权重 $A$ 加权求和。"——**一句话：在"我应该看的位置附近"看几眼，把看到的东西按重要性加权汇总。**

### 3.2 Spatial Cross-Attention 投影数学

对第 $q$ 个 BEV Query 在位置 $p_q = (x_q, y_q)$，其 3D 参考点为：

$$p_q^{3D} = (x_q, y_q, z_j), \quad z_j \in \{-5, -2.33, 0.33, 3.0\} \text{ (4 heights)}$$

投影到第 $c$ 个相机：

$$p_q^{2D, c} = \pi_c(p_q^{3D}) = K_c \cdot [R_c|t_c] \cdot [x_q, y_q, z_j, 1]^T$$

其中 $[R_c|t_c]$ 是相机外参，$K_c$ 是内参矩阵。仅当 $p_q^{2D,c}$ 落在图像 $[0, W_c] \times [0, H_c]$ 内时才参与采样。

> [!note] 为什么要有 4 个高度（-5m 到 3m）？
> 一个 BEV 格子的地面坐标确定，但**物体有多高不知道**——行人 1.7m、货车 3m、立交桥 5m。4 个高度锚点让 query 能"分层看"：网络自己决定哪层有东西。**这是 BEV（2D）应对"高度"问题的经典技巧**（[[3D视觉与投影几何]] 里提过）。

```python
# 投影的关键代码片段（简化自 mmdet3d）
def project_3d_to_2d(points_3d, cam_intrinsics, cam_extrinsics):
    # points_3d: [N, 3] (x, y, z in ego frame)
    # cam_extrinsics: [4, 4] ego→cam transform
    # cam_intrinsics: [3, 3]
    points_cam = cam_extrinsics[:3, :3] @ points_3d.T + cam_extrinsics[:3, 3:]
    points_2d_h = cam_intrinsics @ points_cam  # [3, N]
    points_2d = points_2d_h[:2] / points_2d_h[2:]  # 透视除法
    return points_2d.T  # [N, 2] (u, v)
```

### 3.3 Temporal Self-Attention — Ego-Motion 对齐

关键问题：$t-1$ 时刻的 BEV 特征 $B_{t-1}$ 在 $t-1$ 时刻的**自车坐标系**下，需要对齐到 $t$ 时刻的自车坐标系。

$$\hat{B}_{t-1}(p) = B_{t-1}(T_{t \to t-1} \cdot p)$$

其中 $T_{t \to t-1} = T_{t-1}^{-1} \cdot T_t$ 是 $t$ 到 $t-1$ 的变换矩阵（只包含 $x, y, yaw$，忽略 roll/pitch/z）：

$$T = \begin{bmatrix} \cos\Delta\theta & -\sin\Delta\theta & 0 & \Delta x \\ \sin\Delta\theta & \cos\Delta\theta & 0 & \Delta y \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

> [!note] 白话解释：为什么 warping 是"反着搬"的？
> 车往前开了 2 米，上一帧记录"前方 20 米有车"，这一帧那辆车应该在"前方 18 米"。要把上一帧的 BEV 特征搬到当前坐标系，**每个格子要按"车动了多少"反向移动**——这就是 $T_{t \to t-1}$（当前坐标系 → 上一帧坐标系）的作用。搬完之后，两帧的"同一地面位置"才对齐，才能做时序注意力。

**对齐失败场景**：
- 自车过减速带/上下坡 → z 轴变化大 → warp 不准
- 急转弯 → 仅用 yaw 的刚体变换不够 → 需要完整的 SE(3) 变换（但计算量大）
- 高速场景 → 两帧之间位移大 → warp 后有较大"空白"区域

---

## 四、训练细节

### 4.1 训练配置 (直接来自论文 + 开源代码)

| 项目 | BEVFormer-S (小) | BEVFormer-B (大) |
|------|-----------------|-----------------|
| **Backbone** | ResNet-50 | VoVNet-99 |
| **预训练** | ImageNet | dd3d (深度预训练) |
| **Epoch** | 24 | 60 (w/ CBGS) |
| **Batch Size** | 8 (8 GPUs × 1) | 8 |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999) | 同 |
| **LR** | 2×10⁻⁴ | 2×10⁻⁴ |
| **LR Schedule** | Cosine Annealing | Cosine (warmup 500 steps) |
| **Weight Decay** | 0.01 | 0.01 |
| **Gradient Clip** | max_norm=5 | max_norm=5 |
| **图像分辨率** | 1600×900 → resize 640×960 (train) | 同 |
| **数据增强** | RandomHorizontalFlip, RandomRotation(±22.5°) | + CBGS |
| **训练时间** | ~28h (8×A100) | ~70h |
| **FP16** | 是 | 是 |

> [!note] 从配置里读出两个"经验值"
> - **LR=2×10⁻⁴ + AdamW + Cosine + warmup**：和 [[模型基础知识补充]] 推荐的 BEV 标准配置完全一致——这不是巧合，BEVFormer 就是这套配置的"定义者"之一。
> - **VoVNet-99 预训练用 dd3d（深度预训练）**：在 ImageNet 之外，用带深度的预训练任务（dd3d 是单目 3D 检测预训练）——**预训练任务越贴近下游，迁移效果越好**（呼应 [[模型基础知识补充]] 的预训练策略）。

### 4.2 为什么用 CBGS (Class-Balanced Grouping and Sampling)？

nuScenes 数据集中类别极端不平衡：70% 是 car，bicycle/motorcycle 不到 3%。CBGS 操作：

1. 将训练样本按包含的目标类别分组
2. 每次采样时，从各组均匀采样 → 确保每个 batch 包含稀有类别
3. 效果：稀有类别 (bicycle) AP 提升 5-8%，常见类别不降

> [!note] 直觉：让每个 batch 都"物种多样"
> 如果随机采样，绝大多数 batch 全是 car，bicycle 几个月都见不到一次。CBGS 按类别分组采样，**保证每个 batch 里都有稀有类别**——模型每步都能见到"自行车长什么样"。

### 4.3 损失函数

```python
# 分类损失: Focal Loss
L_cls = -α_t * (1 - p_t)^γ * log(p_t)
# α=0.25, γ=2.0

# 回归损失: L1 Loss
L_reg = Σ |b_pred - b_gt|  # 对 (x,y,z,w,l,h,sinθ,cosθ,vx,vy)

# 总损失
L = L_cls + λ * L_reg  # λ = 2.0 (来自mmdet3d默认)
```

> [!note] 为什么分类用 Focal、回归用 L1？
> BEV 上 40000 个格子只有几十个有目标（正负样本比 1:1000 级别）——**Focal Loss 就是为这种极端不平衡设计的**（[[模型基础知识补充]] 讲过）；回归用 L1 简单直接，注意朝向用 sinθ/cosθ 编码（避免角度 0°/360° 的跳变问题）。

---

## 五、实验数据与消融

### 5.1 主要结果 (nuScenes val set)

| 模型 | Backbone | NDS | mAP | mATE↓ | FPS | 参数量 |
|------|----------|-----|-----|-------|-----|--------|
| BEVFormer-S | R50 | 49.6 | 37.2 | 0.671 | 4.5 | 44.8M |
| BEVFormer-B | R101 | 51.7 | 41.6 | 0.632 | 2.9 | 71.3M |
| BEVFormer-B | VoV-99 | 56.9 | 48.1 | 0.582 | 2.5 | 68.4M |

### 5.2 时序消融实验 (最关键的 ablation！)

| 时序帧数 | NDS | mAP | ΔNDS |
|---------|-----|-----|------|
| 无时序 (0) | 46.0 | 32.1 | — |
| 1 帧历史 | 49.6 | 37.2 | +3.6 |
| 2 帧历史 | 50.2 | 38.0 | +4.2 |
| 3 帧历史 | 50.5 | 38.3 | +4.5 |

> [!question] 面试追问：为什么 3 帧比 1 帧只提升 0.9 NDS？
> 因为 3 帧前的信息已经通过 ego-motion warping 退化太多（车一直在动，历史帧越老，对齐误差越大、遮挡越严重），信息增量有限。**权衡效率 → 只用 1 帧历史。** 这个"边际收益递减"的观察是时序建模的通用规律（后续 StreamPETR 的流式设计也是为此）。

### 5.3 各模块消融

| 配置 | NDS | Δ |
|------|-----|---|
| Full BEVFormer | 56.9 | — |
| 去掉 Temporal SA | 51.8 | -5.1 |
| 去掉 Spatial CA | 31.2 | -25.7 |
| 去掉多尺度特征 | 54.3 | -2.6 |
| 去掉 FOV mask | 52.1 | -4.8 |

> [!warning] 读消融的正确姿势
> **去掉 Spatial CA 掉 25.7 个点**——空间注意力是绝对主力（从图像拿信息）；去掉 Temporal SA 掉 5.1——时序是重要加成；FOV mask 掉 4.8——"只在相机看得见的区域采样"这个细节竟也值 5 个点！**消融表告诉你：模型的核心贡献在哪、哪些细节也别忽视。**

### 5.4 各类别 AP (BEVFormer-B, VoV-99)

| 类别 | AP | 类别 | AP |
|------|-----|------|-----|
| car | 66.0 | bicycle | 42.8 |
| truck | 39.7 | motorcycle | 49.0 |
| bus | 54.2 | traffic_cone | 62.4 |
| trailer | 25.1 | barrier | 55.3 |
| construction | 21.9 | pedestrian | 48.3 |

> [!question] 追问：trailer 和 construction 为什么这么低？
> trailer 形状多变 (挂车有长有短)，construction 类内差异极大 (施工锥、围栏、挖掘机...) → 传统 BBox 表示不适合这些类别 → 需要 Occupancy Network。**这就是 BEV 检测 → 占据网络演进动机的数据证据**（[[占据网络与GOD]] 会展开）。

---

## 六、代码实现走读

### 6.1 代码架构 (mmdet3d/projects/BEVFormer)

```
bevformer/
├── bevformer_head.py           # 主 Head: BEV queries + encoder forward
├── spatial_cross_attention.py  # Deformable Cross-Attention 实现
├── temporal_self_attention.py  # Temporal Self-Attention + ego-motion warp
├── transformer.py              # BEVFormerEncoder 层组装
└── utils.py                    # 坐标变换工具
```

### 6.2 BEV Queries 初始化

```python
# bevformer_head.py (简化)
class BEVFormerHead:
    def __init__(self, bev_h=200, bev_w=200, embed_dims=256):
        # BEV 网格的物理坐标
        xs = torch.linspace(-51.2, 51.2, bev_w)  # [200]
        ys = torch.linspace(-51.2, 51.2, bev_h)  # [200]
        self.bev_positions = torch.stack(torch.meshgrid(xs, ys), dim=-1)  # [200, 200, 2]

        # 可学习的 BEV Query Embedding
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        # Shape: [40000, 256] — 每个网格一个 query

    def get_bev_queries(self, batch_size):
        return self.bev_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        # [B, 40000, 256]
```

> [!note] 注意区分两个东西
> - **bev_positions**：每个格子的**物理坐标** (x, y)——告诉模型"这个格子在地面哪里"（用于投影）。
> - **bev_embedding**：每个格子的**可学习向量**——模型学出来的"这个格子问什么问题"。
> 一个管"位置"，一个管"问题"，两者相加才是完整的 BEV query。

### 6.3 Spatial Cross-Attention 完整流程

```python
# 每个 BEV Query 采样 4 个高度 × 4 个采样点 = 最多 16 个 3D 参考点
# → 投影到 6 个相机 → 每个 3D 点可能落在 0-N_cam 个相机视野内

def spatial_cross_attention(bev_query, bev_pos, img_feats, cam_params):
    """
    bev_query:  [B, N_q, 256]  N_q = 40000
    bev_pos:    [B, N_q, 2]    (x, y) in ego frame
    img_feats:  list of [B, C, H_l, W_l] for l=1..4 (multi-scale)
    cam_params: 每相机的内参 K [3,3] + 外参 [R|t] [4,4]
    """
    B = bev_query.shape[0]

    # 1. 预测采样偏移和注意力权重
    sampling_offsets = self.sampling_offset_linear(bev_query)
    # [B, N_q, n_heads, n_levels, n_points, 2]
    attention_weights = self.attention_weight_linear(bev_query)
    # [B, N_q, n_heads, n_levels, n_points]
    attention_weights = F.softmax(attention_weights, dim=-1)

    # 2. 生成 3D 参考点
    # 对每个 BEV pos (x,y), 扩展 n_height 个 z 高度
    reference_points_3d = torch.stack([
        torch.cat([bev_pos, z.expand_as(bev_pos[..., :1])], dim=-1)
        for z in [-5.0, -2.33, 0.33, 3.0]  # 4 heights
    ], dim=2)  # [B, N_q, n_height, 3]

    # 3. 投影到各相机 + 可变形采样
    # 对每个相机:
    for cam_idx in range(6):
        K = cam_params[cam_idx]['intrinsics']      # [3, 3]
        RT = cam_params[cam_idx]['extrinsics']      # [4, 4]

        # 3D → 2D 投影
        ref_pts_2d = project_to_2d(reference_points_3d, K, RT)
        # [B, N_q, n_height, 2]  (u, v)

        # 检查投影点是否在图像内 (FOV mask)
        valid_mask = (ref_pts_2d[..., 0] >= 0) & (ref_pts_2d[..., 0] < W) & \
                     (ref_pts_2d[..., 1] >= 0) & (ref_pts_2d[..., 1] < H)

        # Deformable Attention 采样
        # 实际偏移 = reference_points_2d + sampling_offsets
        cam_output = deformable_attn_cuda(
            bev_query, ref_pts_2d, sampling_offsets,
            attention_weights, img_feats, valid_mask
        )

    return aggregated_output  # [B, N_q, 256]
```

> [!note] 代码流程对应论文四步
> **① 预测偏移和权重**（网络学"去哪看、看多重要"）→ **② 生成 3D 参考点**（BEV 坐标 + 4 个高度）→ **③ 投影到相机 + FOV 检查**（只有落在图像内的点才算数）→ **④ 可变形采样**（在投影点附近按偏移采样特征）。代码和公式一一对应，读代码就是读公式。

### 6.4 Temporal Self-Attention

```python
def temporal_self_attention(bev_query, prev_bev, ego_motion):
    """
    prev_bev: [B, N_q, 256]  — 上一帧的 BEV 特征
    ego_motion: t → t-1 的变换矩阵 [B, 4, 4] (Δx, Δy, Δyaw)
    """
    # 1. 将上一帧 BEV 特征 warped 到当前坐标系
    # BEV 坐标: [200, 200] → 物理坐标 [-51.2, 51.2]
    prev_pose = self.get_bev_positions()  # [N_q, 2]

    # 应用 ego-motion 逆变换 (把 t 坐标系的点变到 t-1 坐标系)
    # 注意: 坐标系变换方向!
    T = ego_motion  # t 坐标系 → t-1 坐标系的变换
    prev_pose_homo = torch.cat([prev_pose, torch.ones(N_q, 1)], dim=-1)
    warped_pose = (T @ prev_pose_homo.T).T  # [N_q, 3]
    warped_pose = warped_pose[..., :2] / warped_pose[..., 2:]

    # 2. 用 warped pose 在 prev_bev 上 grid_sample
    # grid_sample 期望坐标在 [-1, 1] 范围
    grid = warped_pose / 51.2  # normalize to [-1, 1]
    aligned_prev_bev = F.grid_sample(
        prev_bev.reshape(B, 200, 200, 256).permute(0,3,1,2),
        grid.reshape(B, N_q, 1, 2),
        mode='bilinear', padding_mode='zeros'
    )
    # [B, 256, N_q, 1] → squeeze → [B, N_q, 256]

    # 3. Self-Attention: Q=当前, K,V=对齐后的历史
    # 标准 Multi-Head Self-Attention (不是 Deformable — 注意这里!)
    # 因为 BEV query 已经对齐，直接用标准 Attention
    Q = self.q_proj(bev_query)     # [B, N_q, 256]
    K = self.k_proj(aligned_prev_bev)
    V = self.v_proj(aligned_prev_bev)

    attn_output = F.scaled_dot_product_attention(Q, K, V)
    # [B, N_q, 256]

    return attn_output
```

> [!warning] 代码里最容易被忽略的细节
> **Temporal Self-Attention 用的是"标准 Attention"，不是 Deformable！** 因为 BEV query 和 warped 历史 BEV 是**同构的网格**（1 对 1 对应），不需要稀疏采样。而 Spatial Cross-Attention 里 query 和图像特征"形状不同、关系未知"，才需要 Deformable。**"两种注意力各用其适"是读懂代码的关键。**

---

## 七、面试官深度追问

> 以下模拟面试中面试官的连续追问。能流畅回答这些问题的，才算是真正理解了 BEVFormer。

### Q1: Deformable Attention 为什么 K=4 就够了？

**回答**: 因为 K 个采样点不是在全局随机采样的，而是在**参考点周围**学习偏移。参考点是对 3D 位置的最佳猜测（通过相机内外参投影），K 个采样点在参考点附近精细调整，捕获局部特征差异。消融实验证明 K=4→8 精度提升 <0.5% 但计算量翻倍，所以 K=4 是最优的 cost/accuracy trade-off。

### Q2: 如果有个相机完全黑了（故障/强光），BEVFormer 会怎样？

**回答**: FOV mask 机制会自动跳过该相机。由于 Temporal Self-Attention 融合了历史 BEV 特征，对静态物体（建筑、车道线）影响小（-5%）。但对动态物体（行人、车），历史信息最多补偿 1-2 帧，超过 2 帧该相机持续不可用 → 对应区域 NDS 下降 15-20%。这是 BEVFormer 相比纯 LiDAR 方案的弱点。

### Q3: Temporal Self-Attention 用的是标准 Attention 还是 Deformable Attention？为什么？

**回答**: **标准 Multi-Head Self-Attention**。因为 BEV Query 的时序比较是同位置的 Query 互相 attend（K,V 来自 warped 的上帧 BEV），这是 1-to-1 对应，不需要 deformable。复杂度 $O(N_q^2) = O(40000^2)$ 在 8 头 attention 的优化下是可以接受的，因为每层的 Self-Attention 只做一次（不像 Spatial CA 要做 6 相机 × 4 层级）。

### Q4: ego-motion 对齐只用了 yaw，实际驾驶中有 roll/pitch，为什么可以忽略？

**回答**: 三个原因：
1. **频率**: 正常驾驶中 roll/pitch 变化频率远低于 yaw (转向)，大部分时间接近 0
2. **幅度**: 平路上的 roll/pitch 通常 <2°，对 200×200 BEV grid 的影响 <1 pixel
3. **偏移容错**: Temporal Self-Attention 中的 softmax 会给对齐不完美的区域较低的 attention weight，自适应地降低错误对齐的影响
但**过减速带/上下坡时确实会失效** → 这是已知局限。

### Q5: 多尺度 Deformable Attention 中不同 level 的特征贡献有什么不同？

**回答**: 低层级特征（high resolution, level 1）负责细粒度定位（物体的精确位置、边界），高层级特征（low resolution, level 4）负责语义理解（这个 blob 是车还是树）。Attention weights 自动学习如何组合：对于大物体（truck/bus），高层级特征权重更高；对于小物体（pedestrian/cone），低层级权重更高。

### Q6: 如果 BEV 网格从 200×200 改成 100×100，精度损失多少？为什么？

**回答**: 从论文的 scale ablation 可以估算：
- 200×200 (0.512m/cell): NDS=56.9
- 100×100 (1.024m/cell): NDS 预计下降 3-5 点
原因：
1. 小物体（pedestrian ~0.5m 宽）在 1m grid 中只占 1 cell → 定位模糊
2. 密集场景（停车场）多物体挤在少数 cell 中 → 检测召回下降
3. 但速度从 2.5 FPS 提升到 10+ FPS → 实时部署时这是合理的 trade-off

### Q7: BEVFormer 最大的工程落地难点？

**回答**:
1. **Deformable Attention CUDA Kernel**: 实现复杂，不同 GPU 架构（V100/A100/T4）需要不同的编译优化 → mmdet3d 用了大量 custom CUDA ops
2. **多帧显存**: 需要存储上一帧 BEV 特征 (200×200×256×2bytes = 10MB)，在 6 帧时序下积少成多
3. **低 FPS**: 2.5 FPS 无法满足实时性要求 (10+ FPS) → 需要模型轻量化或 TensorRT 优化（但有 deformable attn 的 TRT 支持不完善）
4. **相机标定漂移**: 实车运行中相机外参会因温度/振动漂移 → 投影不准 → Spatial CA 失效 → 需要在线标定补偿

### Q8: 从 2024 年的视角看，BEVFormer 还有必要学吗？

**回答**: **绝对有必要 — 作为基础范式理解**。
- 它的 Cross-Attention + Temporal Fusion 思想被几乎所有后续工作继承（UniAD, VAD, StreamPETR, BEVFormerv2）
- 即使最终部署用的可能是 BEVDet（更快），面试一定会问 BEVFormer 的原理
- 理解了 BEVFormer 的局限（速度、显存、标定敏感）→ 才能理解为什么后来者做那些改进
- 最新工作如 SparseBEV、FlashOcc 都是在 BEVFormer 基础上的稀疏化/加速

---

## 八、BEVFormer v2 的改进

| 维度 | BEVFormer v1 | BEVFormer v2 |
|------|-------------|-------------|
| **监督信号** | 仅 3D box supervision | + **Perspective Supervision** (3D box 投影到 2D 做辅助 loss) |
| **Backbone** | VoVNet-99 | ConvNeXt-B |
| **任务** | 3D 检测 | 3D 检测 + BEV 分割 |
| **NDS** | 56.9 | **62.8** (+5.9) |
| **mAP** | 48.1 | **53.5** (+5.4) |
| **训练策略** | 单阶段 | 两阶段 (先训分割再训检测) |

**核心改进**：Perspective Supervision 利用 2D 标注（比 3D 标注便宜！）提供了额外的监督信号，让模型明确学习"这个 BEV 位置投影到图像上应该是什么物体"。

> [!note] Perspective Supervision 的直觉
> BEV 检测只用 3D 标注监督（贵、稀疏），v2 把 3D 框投影回 2D 图像上，用 2D 检测/分割的结果做辅助监督（便宜、稠密）。**"3D 真值贵，2D 真值便宜——那就用便宜的真值教 3D 模型"**——和 BEVDepth 用 LiDAR 教深度是同一类"监督迁移"思想。

---

## 九、总结 — BEVFormer 的核心贡献

1. **范式定义**: 首次系统性地提出用 Transformer Query 机制统一多相机 BEV 感知
2. **时空融合**: Temporal Self-Attention + Spatial Cross-Attention → 后续工作的标准模板
3. **工程验证**: 在 nuScenes 上 SOTA，且开源代码 → 学术和工业界的共同基准
4. **局限**: 速度慢（2.5 FPS）、对相机标定敏感、需要存储历史 BEV 特征

---

## 十、常见误区速查

> [!warning] 新手最常踩的坑
> 1. **以为 BEV Query 是"图像特征"**——不是！BEV Query 是**可学习的向量**（Embedding），先天地图位置信息，再通过注意力从图像里"填内容"。
> 2. **以为两个注意力都是 Deformable**——只有 Spatial Cross-Attention 是 Deformable，Temporal Self-Attention 是标准 Attention（Q6 讲过）。
> 3. **忽略 FOV mask**——投影到图像外的参考点必须被 mask 掉（消融显示它值 4.8 个 NDS 点）。
> 4. **以为时序帧数越多越好**——消融显示 3 帧比 1 帧只多 0.9 NDS，历史越老 warp 退化越严重（边际收益递减）。
> 5. **以为 BEVFormer 是"纯视觉魔法"**——它高度依赖相机标定（外参投影），标定漂移是量产最大痛点之一。

---

## ✅ 检验自己（自测题）

> [!question] Q1：为什么 BEVFormer 用 Query 机制而不是把图像特征直接投影到 BEV？用一句话回答本质。
> 提示：想想单目相机的深度问题。

> [!success]- 参考答案
> 因为单目相机每个像素只知道"一条射线"而不知道确切深度，无法精确投影；Query 机制让网络通过注意力**自己学"该去图像的哪个位置看"**，用隐式方式解决深度问题，避免了显式深度估计的级联误差。

> [!question] Q2：Spatial Cross-Attention 和 Temporal Self-Attention 各回答什么问题？在 Encoder 里的顺序是什么？
> 提示：一个管"现在"，一个管"过去"。

> [!success]- 参考答案
> Spatial CA 回答"当前图像上这个 BEV 位置有什么"（从多相机特征采样）；Temporal SA 回答"过去这个位置有什么"（融合历史 BEV 特征，需先 ego-motion 对齐）。顺序是先时序后空间：先和历史特征对齐融合获得延续性，再去图像补充当前证据，最后 FFN 消化。6 层重复这个循环。

> [!question] Q3：Deformable Attention 的复杂度是多少？相比标准 Attention 省在哪？
> 提示：N vs N×K。

> [!success]- 参考答案
> O(N·L·K·C)，N=40000 个 query、L=4 层特征、K=4 个采样点 → 约 1.6 亿次运算；标准 Attention 是 O(N²·C) ≈ 4000 亿次，差约 2500 倍。省在"每个 query 只采样 K 个点"而不是和所有位置全连接——采样点在参考点附近由网络学习偏移，兼顾效率与精度。

> [!question] Q4：时序对齐为什么要"反向 warp"？只对齐 x/y/yaw 可行吗，有什么失败场景？
> 提示：车动了，世界没动；想想减速带和急转弯。

> [!success]- 参考答案
> 上一帧的 BEV 特征定义在上一帧的自车坐标系里，要搬到当前坐标系，需要把每个格子按"车动了多少"反向移动（T_{t→t-1}）。只对齐 x/y/yaw 在平路正常驾驶时可行（roll/pitch <2° 影响 <1 像素），但过减速带/上下坡（z 变化大）、急转弯（需要完整 SE(3)）时会 warp 不准——这是已知局限，也是后续工作改进的方向之一。

> [!question] Q5：消融表里"去掉 Spatial CA"掉了 25.7 个点、"去掉 Temporal SA"掉了 5.1 个点，这说明了什么？
> 提示：核心 vs 加成。

> [!success]- 参考答案
> 说明空间注意力是 BEVFormer 的绝对核心（从图像获取信息的通道，去掉几乎等于没感知），时序注意力是重要加成（+5 点）但不是根基。还说明 FOV mask（4.8 点）和多尺度（2.6 点）等"细节"也不容忽视——消融表是判断"模型真正的贡献点"的最可靠证据。

> [!question] Q6：BEVFormer 相比 BEVDet 在部署上有什么劣势？为什么量产反而常选 BEVDet？
> 提示：算子、工具链、量化、FPS。

> [!success]- 参考答案
> ① Deformable Attention 需要自定义 CUDA kernel，TensorRT 支持不完善（要写 plugin）；② 低 FPS（2.5 vs 15+）无法满足实时性；③ INT8 量化精度损失大（2-3% vs <0.5%）；④ 对相机标定敏感（标定漂移直接导致投影失效）。量产选 BEVDet 是因为"精度略低但能跑起来"，精度差距再用 BEVDepth 的深度监督补齐——部署约束优先于精度追求。

---

## 🛠 动手练习

### 练习 1：手算一次"参考点投影 + 采样"（20 分钟）

给定一个 BEV query 位置 p = (20, 3)（前方 20 米、偏右 3 米），高度 z=1m，相机内参 K=[1000,0,640; 0,1000,360; 0,0,1]，外参为单位变换（相机在世界原点朝前）。
1. 写出 3D 参考点 (20, 3, 1)。
2. 用 [[3D视觉与投影几何]] 的投影公式算像素坐标 (u, v)。
3. 假设图像是 1280×720，判断该点是否落在图像内（FOV mask）。
4. 如果落在图像内，这个 BEV query 会从哪个相机采样？

> [!tip] 做完后自问
> 如果把高度改成 z=-5m（地下），投影点会怎么变？这解释了"4 个高度锚点"为什么能覆盖从地下到高空的物体。

### 练习 2：读代码 + 画数据流（60-90 分钟）

打开 BEVFormer 官方仓库（GitHub: fundamentalvision/BEVFormer），读 `projects/mmdet3d_plugin/bevformer/` 下的 `spatial_cross_attention.py` 和 `temporal_self_attention.py`：
1. 在代码里找到本文 6.3/6.4 节的每一步（预测偏移、3D 参考点、投影、FOV mask、采样）。
2. 画出完整的张量形状流（从 [B, 40000, 256] 的 query 到输出）。
3. 回答：`sampling_offsets` 的形状为什么是 [B, N_q, n_heads, n_levels, n_points, 2]？最后那个 2 是什么？

> [!tip] 读代码的姿势
> **不要逐行读，先找"形状"**——每个中间张量的形状就是它的"身份"，形状看懂了，代码逻辑就懂了一大半。

### 练习 3：跑通 BEVFormer 推理（可选，1-2 天）

按官方仓库在 nuScenes mini 上跑通推理（参照 [[学习计划]] Phase 6）：
1. 下载 nuScenes mini + 官方权重。
2. 跑一次推理，可视化 BEV 检测结果。
3. 尝试修改 `bev_h/bev_w` 为 100×100，观察精度（NDS/mAP）变化——验证 Q6 的答案。

> [!warning] 环境提示
> BEVFormer 对 PyTorch/CUDA/mmdet3d 版本要求严格（老代码 + 新环境是地狱级配置），建议用官方提供的 Docker 或 conda 环境。**环境配置本身就是学习的一部分**——[[训练排错实战手册]] 有排查思路。

---

## ➡️ 下一步学什么

按知识库学习路径，读完本篇你应该接着：

1. **[[多传感器融合基础]]** —— 学习路径第 10 站：BEV 之后读融合（理解"BEV 是多模态融合的公共坐标系"）。
2. **[[占据网络与GOD]]** —— 学习路径第 11 站：从 BEV 检测升级到 3D 占据（BEVFormer 的 trailer/construction 检测短板 → 占据网络的动机）。
3. **[[PETR系列]]** —— 进阶：第三条路线，用 3D 位置编码替代"投影 + 采样"。
4. **[[UniAD详解]]** —— 端到端：BEVFormer 的 Query 思想如何被 UniAD 发扬光大（多任务 Query 交互）。

> 💡 带着这张"时空注意力"图去读 [[UniAD详解]]——你会发现 UniAD 的 Track Query 机制就是 BEVFormer Temporal Attention 思想的端到端延伸。

---

## 相关笔记

- [[Transformer架构详解]] — Attention 机制基础
- [[Transformer进阶知识]] — FlashAttention、RoPE 等进阶
- [[BEV感知全景]] — BEV 四路线总览
- [[BEVDet与BEVDepth]] — LSS 派对照
- [[PETR系列]] — 3D 位置编码路线
- [[占据网络与GOD]] — BEV → 3D 占据升级
- [[UniAD详解]] — Query 思想的端到端延伸
- [[常见面试题-感知算法]] — 面试实战
