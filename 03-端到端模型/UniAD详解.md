---
tags: [end-to-end, planning, model]
created: "2026-07-21"
---

# UniAD 详解

> **论文**: Planning-oriented Autonomous Driving (CVPR 2023 Best Paper Award)
> **机构**: 上海 AI Lab / 武汉大学 / 商汤科技
> **代码**: [GitHub](https://github.com/OpenDriveLab/UniAD)

---

## 一、核心思想

**以规划为导向** (Planning-oriented)：不再把检测/跟踪/预测/规划割裂开来，而是建立一个统一的端到端框架，让所有任务联合优化。

```
传统 Pipeline:   检测 → 跟踪 → 预测 → 规划  (各自独立)
UniAD:           所有任务通过 Query 统一交互 + 联合优化
```

**关键哲学**：检测和跟踪不只是"输出结果"，它们的中间特征应该为规划服务。

---

## 二、整体架构

```
┌────────────────────────────────────────────────────┐
│                     UniAD                          │
│                                                    │
│  多相机图像                                         │
│      ↓                                             │
│  ┌──────────────────┐                              │
│  │ BEV Encoder      │  ← 基于 BEVFormer           │
│  │ (特征提取)       │                              │
│  └────────┬─────────┘                              │
│           ↓                                        │
│  ┌──────────────────┐    ┌──────────────────┐     │
│  │ TrackFormer      │ ←→ │ MapFormer        │     │
│  │ (检测+跟踪)      │    │ (在线建图)       │     │
│  └────────┬─────────┘    └────────┬─────────┘     │
│           │  Track Query          │  Map Query     │
│           └──────────┬───────────┘                │
│                      ↓                             │
│  ┌──────────────────────────────────────┐         │
│  │ MotionFormer (轨迹预测)              │         │
│  │  输入: Track Query + Map Query + 自车状态│     │
│  │  输出: 多模态未来轨迹                 │         │
│  └──────────────────┬───────────────────┘         │
│                     ↓                              │
│  ┌──────────────────────────────────────┐         │
│  │ OccFormer (占据预测)                  │         │
│  │  预测未来的占据栅格变化               │         │
│  └──────────────────┬───────────────────┘         │
│                     ↓                              │
│  ┌──────────────────────────────────────┐         │
│  │ Planner (规划器)                      │         │
│  │  输入: 轨迹预测 + 占据预测 + 导航信息  │         │
│  │  输出: 自车规划轨迹                   │         │
│  └──────────────────────────────────────┘         │
└────────────────────────────────────────────────────┘
```

---

## 三、各模块详解

### 1. BEV Encoder (特征提取)

基于 BEVFormer，将多相机图像编码为 BEV 特征。

```
6 张图像 → ResNet-101 → FPN → BEVFormer Encoder → BEV Features B ∈ R^{200×200×256}
```

### 2. TrackFormer (检测 + 跟踪)

**统一检测和跟踪**：不再先检测再匹配。

```
Detect Queries  → 当前帧检测结果
Track Queries   → 携带历史信息的跟踪查询
                     ↓
              将当前检测与历史 Track Queries 关联
                     ↓
               更新 Track Queries (检测+跟踪一体化)
```

**Track Query** 包含：
- Agent 的位置和运动状态
- Agent 的外观特征
- 历史轨迹信息

### 3. MapFormer (在线建图)

从 BEV 特征中预测车道线、人行横道、可行驶区域等地图元素。

```
BEV Features → Map Queries (可学习的查询) → 地图元素 (折线/多边形)
```

**在线建图的意义**：不依赖高精地图，实时感知道路结构。

### 4. MotionFormer (轨迹预测)

**输入**：
- Track Queries（其他 agent 的当前+历史状态）
- Map Queries（道路结构信息）
- Ego State（自车状态）

**架构**：Agent-Agent + Agent-Map + Agent-Goal 三层交互

```
Mode Queries → Agent-Level Interaction → Map-Level Interaction → Goal-Level Interaction → 多模态轨迹
```

### 5. OccFormer (占据预测)

**为什么要 OccFormer**：轨迹预测只能处理被跟踪的 agent，占据预测可以覆盖所有未被跟踪的空间（如散落货物、异常路障）。

```
Track Queries + Map Queries → OccFormer → 未来占据栅格预测
```

### 6. Planner (规划器)

```
规划输入:
  ├── 未来轨迹预测 (MotionFormer)
  ├── 未来占据预测 (OccFormer)
  ├── 导航信号 (目标点)
  └── 自车历史轨迹

               ↓
         规划器 (基于 Attention 的规划)
               ↓
         自车未来轨迹 (waypoints)
```

**损失函数**：
```
L_plan = L2(预测轨迹, 真值轨迹) + Collision Loss + Direction Loss
```

**Collision Loss**：鼓励规划轨迹远离占据预测中的占据区域。

---

## 四、训练策略

### 两阶段训练

```
Stage 1: 预训练感知模块
  BEV Encoder + TrackFormer + MapFormer + MotionFormer + OccFormer
  各自独立训练

Stage 2: 端到端联合微调
  加载 Stage 1 权重
  所有模块 + Planner 联合优化
```

### 损失函数

```
总损失 = L_track + L_map + L_motion + L_occ + L_plan

每个模块的损失都有权重系数
```

---

## 五、优缺点分析

### 优点

- ✅ CVPR 2023 **最佳论文**
- ✅ 首个真正以规划为导向的端到端框架
- ✅ 模块间通过 Query 灵活交互，不强制固定接口
- ✅ 同时输出检测、跟踪、预测、规划，中间过程可解释
- ✅ 联合优化避免了模块间误差累积

### 缺点

- ❌ **计算量巨大**（多个 Transformer 模块堆叠）
- ❌ 训练复杂，需要多阶段预训练
- ❌ 模型过大，实时部署困难
- ❌ 只在 nuScenes 上验证，泛化性待考察
- ❌ 开环评测（nuScenes），闭环效果未知

---

## 六、关键数据

| 指标 | 数值 |
|------|------|
| **输入分辨率** | 1600×900 |
| **BEV 尺寸** | 200×200 (nuScenes) |
| **感知范围** | 51.2m × 51.2m (前60m, 后30m) |
| **FPS** | ~1.8 FPS (V100) |
| **模型大小** | 大（多模块堆叠）|

---

## 七、后续工作

| 工作 | 改进方向 |
|------|----------|
| **VAD** | 矢量化场景表征，提升效率 |
| **GenAD** | 生成式端到端，VAE+规划 |
| **UniAD V2** | 进一步提升效率和性能 |

---

## 相关笔记

- [[端到端自动驾驶概览]]
- [[VAD详解]]
- [[BEVFormer详解]]
- [[占据网络与GOD]]
- [[世界模型]]
