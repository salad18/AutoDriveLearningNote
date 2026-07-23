---
tags: [BEV, detection, LSS]
created: "2026-07-21"
---

# BEVDet 与 BEVDepth

> **核心范式**: Lift-Splat-Shoot (LSS) — 显式估计深度，将 2D 特征投影到 3D → 拍平到 BEV。

---

## 一、LSS 范式回顾

```
Lift:       图像特征 × 深度分布 → 3D 锥体特征 (Frustum Features)
              ↓
Splat:      锥体特征按体素索引散射到 BEV 栅格 (Voxel Pooling)
              ↓
Shoot:      在 BEV 上运行下游检测/分割 Heads
```

**原始 LSS 论文**: Philion & Fidler, ECCV 2020

---

## 二、BEVDet

> 论文: BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View (2022)
> 代码: [GitHub](https://github.com/HuangJunJie2017/BEVDet)

### 整体架构

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐
│ Image-View   │ →  │ View        │ →  │ BEV          │ →  │ Detection  │
│ Encoder      │    │ Transform   │    │ Encoder      │    │ Head       │
│ (Backbone+FPN)│   │ (LSS)       │    │ (BEV Backbone)│   │ (CenterPoint)
└──────────────┘    └─────────────┘    └──────────────┘    └────────────┘
```

### 1. Image-View Encoder

- **Backbone**: ResNet-50 / Swin-T
- **Neck**: FPN → 多尺度特征
- **输出**: 每个相机的 2D 特征图 F ∈ R^{C_f × H × W}

### 2. View Transform (LSS)

```
Step 1: 深度估计
  对特征图每个像素 (u,v):
    特征 f ∈ R^{C_f}
    深度分布 d ∈ R^{D} (通过 1×1 Conv 预测)
    → 锥体特征 = f ⊗ d  (外积 → R^{C_f × D})

Step 2: Voxel Pooling
  锥体特征 → 3D 世界坐标 → BEV 栅格索引 → 累积/平均池化
  → BEV 特征 B ∈ R^{C_f × X × Y}
```

### 3. BEV Encoder

- 在 BEV 特征上运行额外的 CNN/ResNet 层
- 目的：补偿 Voxel Pooling 的信息损失
- 使用 2 倍上采样提高分辨率

### 4. Detection Head

- 基于 CenterPoint：预测 heatmap + 回归属性 (z, w, h, l, θ, v)
- 3D 框解码：heatmap peak → 位置 + 属性 → 完整 3D box

### BEVDet 的工程优化

| 优化 | 方法 |
|------|------|
| **预计算** | 离线计算每个像素对应的 3D 坐标（相机内外参不变时） |
| **间隔帧计算** | 不必每帧都算 LSS，可隔帧计算（前提：帧间变化小） |
| **BEV 特征缓存** | 缓存前几帧的 BEV 特征，相当于免费时序信息 |

---

## 三、BEVDepth

> 论文: BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection (AAAI 2023)

### 核心问题

**LSS 的深度估计不准** → 3D 投影位置错误 → BEV 特征混乱。

### 解决方案：深度真值监督

```
Camera 图像
    ↓
深度估计网络 (DepthNet)  ← LiDAR 点云深度真值监督
    ↓
准确的深度分布 d
    ↓
LSS 投影 → 高质量 BEV 特征
```

### 关键改进

#### 1. Depth Correction（相机参数编码）

```
将相机内参 (fx, fy, cx, cy) 编码后注入深度预测网络
→ 深度预测能感知当前相机的几何特性
→ 多相机可共享同一个深度预测网络
```

#### 2. Depth Supervision（点云深度监督）

```
从 LiDAR 点云投影到图像 → 稀疏深度真值
用这些真值监督深度预测网络
Loss: Binary Cross-Entropy + Depth-aware Loss
```

#### 3. Efficient Voxel Pooling

- 利用 CUDA 实现 GPU 加速的 Voxel Pooling
- BEVDepth-tiny 可以在 Orin 上实时运行

### BEVDepth 结构

```
Image → Backbone + FPN → 特征 F
                              ↓
              ┌────── DepthNet ──────┐
              │   Concat(特征, 相机参数编码) │
              │   → 深度分布 d              │
              │   ← LiDAR 点云监督          │
              └──────────────────────┘
                              ↓
           F × d → Voxel Pooling → BEV Features
                              ↓
                     BEV Encoder → Detection Heads
```

---

## 四、BEVDet4D (时序扩展)

> 在 BEVDet 基础上加入 4D 时序信息（3D 空间 + 时间维度）

### 核心思路

```
1. 将历史帧的 BEV 特征对齐到当前帧
2. 在 BEV 空间做 Concat 或 Temporal Fusion
3. 提供速度和运动信息
```

**速度预测**：时序 BEV 特征天然包含运动信息，可直接预测目标速度。

---

## 五、模型对比

| 特性 | BEVDet | BEVDepth | BEVFormer |
|------|--------|----------|-----------|
| **视图变换** | LSS (显式) | LSS (显式+监督) | Transformer (隐式) |
| **深度来源** | 学习 | 学习 + LiDAR 监督 | 隐式 (Cross-Attn) |
| **时序** | 间隔帧 | ❌ (BEVDet4D 加入) | ✅ Temporal Self-Attn |
| **速度** | 较快 | 快 (CUDA 加速) | 较慢 |
| **部署** | 已落地 | 已落地 (tiny) | 需优化 |
| **可解释性** | 高 | 高 | 低 |

---

## 六、关键概念速记

| 概念 | 含义 |
|------|------|
| **LSS** | Lift-Splat-Shoot，视图变换范式 |
| **Frustum Features** | 锥体特征，2D API × 深度 = 3D 特征 |
| **Voxel Pooling** | 将 3D 特征映射到 BEV 栅格 |
| **Depth Supervision** | 用 LiDAR 点云监督深度预测 |
| **Depth Correction** | 相机参数编码引导深度估计 |
| **BEVDet4D** | 时序扩展版 |

---

## 📖 推荐资料

- LSS 论文 (Philion & Fidler, 2020)
- BEVDet 论文 + 开源代码
- BEVDepth 论文 (AAAI 2023)
- CenterPoint 论文（Detection Head 基础）

---

## 相关笔记

- [[BEV感知全景]]
- [[BEVFormer详解]]
- [[PETR系列]]
- [[3D视觉与投影几何]]
