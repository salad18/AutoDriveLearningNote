---
tags: [BEV, PETR, 3D-detection, position-encoding]
created: "2026-07-21"
---

# PETR 系列

> **核心思想**: 用 3D 位置编码（3D Position Encoding）隐式完成视角变换，省去显式的 LSS 投影或 Cross-Attention 采样。

---

## 一、PETR (ECCV 2022)

> **论文**: PETR: Position Embedding Transformation for Multi-View 3D Object Detection
> **机构**: 旷视科技 (Megvii)
> **代码**: [GitHub](https://github.com/megvii-research/PETR)

### 核心思想

PETR 的关键创新：**3D 坐标生成器 → 3D 位置编码 → 注入 Transformer**。不需要像 BEVFormer 那样做明确的投影采样。

### 架构

```
┌──────────────┐    ┌─────────────────┐    ┌────────────────┐    ┌──────────┐
│ 多相机图像    │ →  │ Backbone        │ →  │ 2D 特征        │    │ 3D 坐标   │
│              │    │ (ResNet-50)   │    │ [HW, C]       │    │ 生成器    │
└──────────────┘    └─────────────────┘    └───────┬────────┘    └─────┬─────┘
                                                   │                   │
                                                   ▼                   ▼
                                          ┌─────────────────────────────┐
                                          │  生成 3D 位置编码            │
                                          │  3D PE = MLP(3D_coord)     │
                                          │  特征 = 2D_feat + 3D_PE    │
                                          └──────────────┬──────────────┘
                                                         ▼
                                          ┌─────────────────────────────┐
                                          │   Transformer Decoder       │
                                          │   Object Queries × Cross-Attn│
                                          │   (Q:Object, K/V:Img+3DPE)  │
                                          └──────────────┬──────────────┘
                                                         ▼
                                                  3D 检测结果
```

### 3D 坐标生成器

对图像特征的每个像素 (u, v)：
1. 在相机视锥中采样离散深度值 (d₁, d₂, ..., d_D)
2. 对应世界坐标点 = Camera.unproject(u, v, d_i)
3. 将这些 3D 点和像素特征绑定

**关键**：网络通过 3D PE 隐式学习"这个像素特征对应 3D 空间中哪个位置"。

### 3D Position Encoding

```
3D PE(p) = σ(MLP(p))
p = (x, y, z) 世界坐标

然后:
特征 = 2D_image_feat + 3D_PE
```

**为什么有效**：Transformer Attention 能看到不同像素的 3D 位置关系，从而隐式完成视角变换。

### Decoder

```
Object Queries (类似 DETR) --- Q
图像特征 + 3D PE ------------ K, V
         ↓
    Cross-Attention
         ↓
    3D BBox Predictions
```

---

## 二、PETRv2 (ICCV 2023)

> 在 PETR 基础上加入**时序建模**和**BEV 分割辅助任务**。

### 核心改进

#### 1. Temporal Modeling（时序建模）

```
当前帧图像 + 历史帧图像
     ↓
共享 Backbone → 2D 特征 + 3D PE
     ↓
加入 Temporal Aligned Position Embedding
→ 编码不同帧之间特征的时间关系
```

**实现**：
- 根据 ego-motion，将历史帧的 3D 位置编码对齐到当前帧
- 在 Transformer 中同时处理当前帧和历史帧的 feature

#### 2. Multi-Task Learning

在 3D 检测基础上添加 **BEV 分割**和 **车道线检测**辅助任务：

```
共享 Encoder → 多任务 Head:
  - 3D 检测 Head
  - BEV 语义分割 Head
  - 车道线分割 Head
```

辅助任务提供更丰富的监督信号，提升特征质量。

#### 3. 特征位置编码

PETRv2 将**相机内参**也编码到位置信息中：
```
PE_enhanced = PE(3D_coord) + PE_camera(K)
```

---

## 三、StreamPETR (ICCV 2023)

> **论文**: StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection

### 核心创新：Object-Centric 时序建模

传统时序建模**以特征为中心**（propagate BEV features），StreamPETR**以目标为中心**（propagate object queries）：

```
过去帧的 Object Queries (包含位置+特征)
         ↓
  Motion Prediction (预测当前帧位置)
         ↓
  与当前帧图像特征做 Cross-Attention
         ↓
  更新 Object Queries
         ↓
  新一帧 3D 检测结果
```

### 优势

- **效率高**：不需要传播整张 BEV 特征图，只传递 object queries (几百个向量)
- **长时序**：轻量级传递，容易扩展到长时序（30+ 帧）
- **速度感知**：Object Query 的时序传播天然包含速度信息

### 架构对比

```
传统时序 BEV:   特征图级别传播 (H×W×C)
StreamPETR:     Object Query 级别传播 (N_obj × C)
```

---

## 四、PETR 系列对比

| 特性 | PETR | PETRv2 | StreamPETR |
|------|------|--------|------------|
| **视图变换** | 3D PE (隐式) | 3D PE (隐式) | 3D PE (隐式) |
| **时序** | ❌ | ✅ (特征级) | ✅ (目标级) |
| **辅助任务** | ❌ | ✅ (分割+车道线) | ❌ |
| **实时性** | 中等 | 中等 | ✅ 高效 |
| **长时序** | ❌ | 有限 | ✅ 30+ 帧 |
| **代码** | ✅ 开源 | ✅ 开源 | ✅ 开源 |

---

## 五、与 BEVFormer 的对比

| 维度 | PETR | BEVFormer |
|------|------|-----------|
| **视图变换** | 3D PE 隐式编码 | Cross-Attention 显式采样 |
| **BEV 表示** | 不显式构建 BEV 特征 | 显式 BEV 网格 (H×W) |
| **投影方式** | 3D → 2D（坐标隐式关联）| 3D reference points → 2D 投影 |
| **计算** | 一次 forward pass | 每层都要做 Cross-Attn |
| **优势** | 简洁、端到端 | BEV 表示对下游友好 |

---

## 📖 推荐资料

- PETR 论文 (ECCV 2022)
- PETRv2 论文 (ICCV 2023)
- StreamPETR 论文 (ICCV 2023)
- DETR 论文 (ECCV 2020) — Object Query 机制基础

---

## 相关笔记

- [[Transformer架构详解]]
- [[BEVFormer详解]]
- [[BEV感知全景]]
- [[端到端自动驾驶概览]]
