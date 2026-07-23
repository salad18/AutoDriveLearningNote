---
tags: [BEV, transformer, model]
created: "2026-07-21"
---

# BEVFormer 详解

> **论文**: BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers (ECCV 2022)
> **机构**: 上海 AI Lab / 上海交大
> **代码**: [GitHub](https://github.com/fundamentalvision/BEVFormer)

---

## 一、核心思想

通过**时空 Transformer** 将多相机 2D 图像特征聚合到统一的 BEV 空间。

```
多相机图像 → Backbone → 图像特征
                              ↓
            BEV Queries ←→ Spatial Cross-Attention (空间)
            BEV Queries ←→ Temporal Self-Attention (时序)
                              ↓
                         BEV 特征 → 检测/分割 Heads
```

**一句话**：用可学习的 BEV Queries，通过 Cross-Attention 去图像上"问"每个位置的视觉特征。

---

## 二、网络结构

### 整体 Pipeline

```
┌──────────────────────────────────────────────────────┐
│                    BEVFormer                         │
│                                                      │
│  6 个 Encoder Layer，每个包含：                       │
│                                                      │
│  ┌─────────────────────┐    ┌──────────────────────┐ │
│  │ Temporal            │    │ Spatial              │ │
│  │ Self-Attention      │ →  │ Cross-Attention      │ │
│  │ (历史 BEV 融合)     │    │ (多相机图像查询)      │ │
│  └─────────────────────┘    └──────────────────────┘ │
│          ↓                          ↓                │
│  ┌─────────────────────────────────────────────┐    │
│  │     FFN + Add & Norm                        │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  输入: BEV Queries (200×200 网格, H×W×C)           │
│        + 历史 BEV 特征                               │
│        + 多视图图像特征 (6 个相机)                    │
│                                                      │
│  输出: 统一的 BEV 特征图                              │
└──────────────────────────────────────────────────────┘
```

### BEV Queries

- **网格定义**：BEV 空间划分为 H×W (如 200×200) 个格子
- **物理范围**：通常 51.2m × 51.2m，分辨率 0.256m
- **每个 Query**：包含位置编码 Q ∈ R^{C}，定义在 (x, y) 的 BEV 坐标
- **Z 轴处理**：每个 Query 在多高度采样（如 z=-1m 到 z=5m）

### Spatial Cross-Attention（空间交叉注意力）

```
1. 每个 BEV Query 在 3D 空间中采样 N_ref 个参考点
   参考点 = (x, y, z_1), (x, y, z_2), ..., (x, y, z_Nref)

2. 将每个 3D 参考点投影到每个相机的图像平面
   IF 投影点落在相机视野内：
     在图像特征图上用 Deformable Attention 采样

3. 所有相机采样的特征加权聚合 → BEV Query 更新
```

**关键设计**：
- 使用 Deformable Attention 减少计算量（每个 query 只采样 K=4 个点）
- 每个 BEV Query 只关注能看到它的相机（FOV mask）

### Temporal Self-Attention（时序自注意力）

```
1. 获取历史 BEV 特征 B_{t-1}
2. 根据 ego-motion 将 B_{t-1} 对齐到当前坐标系
3. 对当前 BEV Query 做 Self-Attention，K/V 来自对齐后的 B_{t-1}
4. 融合时域信息
```

**时序对齐公式**：
```
B'_{t-1} = Warp(B_{t-1}, ego_pose_t^{-1} · ego_pose_{t-1})
```

---

## 三、训练策略

### 损失函数

```
总损失 = L_cls (分类) + L_reg (回归)
回归使用 L1 Loss
分类使用 Focal Loss
```

### 细节

- **Backbone**：ResNet-101 + FPN（或 VoVNet，Swin Transformer）
- **BEV 分辨率**：200×200 (nuScenes)
- **参考高度**：4 个高度层 (-1m, -0.5m, 0m, 0.5m)
- **时序帧数**：1 帧历史（对齐后）

---

## 四、BEVFormer v2

> 论文 (2023)

### 改进点

1. **透视监督 (Perspective Supervision)**：在训练时也加入 2D 检测头，提供额外的 2D 监督信号
2. **更强的 Backbone**：使用 InternImage 等现代 backbone
3. **两阶段训练**：先在透视视角预训练 → 再在 BEV 视角微调

### 效果

nuScenes 3D 检测 NDS 从 v1 的 51.7% 提升至 v2 的约 68%（随 backbone 变化）。

---

## 五、优缺点分析

### 优点

- ✅ BEV 表示天然适合融合多相机和时序信息
- ✅ Deformable Attention 高效，不显式依赖深度估计
- ✅ 端到端可微，特征学习灵活

### 缺点

- ❌ 计算量仍较大（每个 query 都要做 cross-attention）
- ❌ Z 轴高度采样是离散的，对高处物体可能不准
- ❌ 对相机外参标定敏感
- ❌ 实时部署仍需要优化

---

## 六、关键概念

| 概念 | 含义 |
|------|------|
| **BEV Query** | 可学习的 BEV 网格锚点 |
| **Spatial Cross-Attention** | BEV Query 去图像上查询特征 |
| **Temporal Self-Attention** | 融合历史帧的 BEV 特征 |
| **Deformable Attention** | 稀疏采样降低计算量 |
| **Ego-motion Alignment** | 自车运动补偿对齐历史帧 |
| **FOV Mask** | 过滤掉看不到该位置的相机 |

---

## 📖 论文关键信息

- **发表**: ECCV 2022
- **引用**: 1000+
- **数据集**: nuScenes
- **指标**: NDS 51.7%, mAP 41.6% (ResNet-101 baseline)

---

## 相关笔记

- [[BEV感知全景]]
- [[Transformer架构详解]]
- [[PETR系列]]
- [[BEVDet与BEVDepth]]
