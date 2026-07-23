---
tags: [end-to-end, planning, vectorized]
created: "2026-07-21"
---

# VAD 详解

> **论文**: VAD: Vectorized Scene Representation for Efficient Autonomous Driving (ICCV 2023)
> **机构**: 上海 AI Lab / 华中科技大学
> **代码**: [GitHub](https://github.com/hustvl/VAD)

---

## 一、核心思想

**用矢量化场景表征替代密集栅格表征，大幅提升端到端自动驾驶的效率。**

```
UniAD:   dense BEV features → dense Occupancy → Planner  (计算量大)
VAD:     vectorized scene → lightweight Planner          (高效)
```

**为什么矢量化**：
- 密集占据栅格 200×200×16 = 640K 体素，其中大部分是空的
- 矢量化只需要存储关键元素的坐标（agent 轨迹、车道线、边界）
- 矢量化与下游规划天然匹配（规划输出也是轨迹点）

---

## 二、整体架构

```
┌──────────────────────────────────────────────┐
│                    VAD                        │
│                                               │
│  多相机图像                                    │
│      ↓                                        │
│  ┌──────────────────┐                         │
│  │ Image Backbone   │                         │
│  │ (ResNet-50)      │                         │
│  └────────┬─────────┘                         │
│           ↓                                   │
│  ┌──────────────────┐                         │
│  │ BEV Encoder      │  ← BEVFormer light       │
│  └────────┬─────────┘                         │
│           ↓                                   │
│  ┌──────────────────┐    ┌──────────────────┐ │
│  │ Vectorized       │    │ Vectorized       │ │
│  │ Motion (Agent)   │    │ Map (Lane)       │ │
│  │    ↓             │    │    ↓             │ │
│  │ Agent 轨迹 + 状态 │    │ 车道线 + 边界    │ │
│  └────────┬─────────┘    └────────┬─────────┘ │
│           └───────────┬───────────┘           │
│                       ↓                       │
│  ┌──────────────────────────────────┐         │
│  │ Vectorized Scene Tokens           │         │
│  │ [Agent1, Agent2, ..., Lane1, ...] │         │
│  └──────────────┬───────────────────┘         │
│                 ↓                              │
│  ┌──────────────────────────────────┐         │
│  │ Planning Transformer              │         │
│  │  输入: Scene Tokens + Ego State   │         │
│  │  输出: 未来轨迹点 (waypoints)     │         │
│  └──────────────────────────────────┘         │
└──────────────────────────────────────────────┘
```

---

## 三、矢量化的场景表征

### 矢量化 Agent 运动

```
每个 Agent = {
  type: car/pedestrian/cyclist
  position: (x, y)            ← 当前位置
  velocity: (vx, vy)
  heading: θ                  ← 朝向
  size: (w, l)               ← 宽、长
  track_id: int              ← 跟踪 ID
}
```

**优势**：N 个 agent 只需要 N 个向量，而不是 N × 密集栅格。

### 矢量化地图

```
地图元素:
  ├── 车道线 (Lane): polyline → [(x1,y1), (x2,y2), ...]
  ├── 人行横道: polygon
  ├── 可行驶区域边界
  └── 停止线
```

**在线矢量化建图**：
```
BEV Features → Map Query → Vectorized Map Elements
```

使用类似 MapTR / VectorMapNet 的方式，直接从 BEV 特征生成矢量地图。

### 场景 Token 构建

```
所有 Agent 轨迹  ─→ Agent Tokens
所有车道线      ─→ Map Tokens
                   ↓
           Scene Tokens = Concat(Agent Tokens, Map Tokens, Ego Token)
```

---

## 四、规划 Transformer

### 输入

```
[Scene Tokens, Ego State, Navigation Goal]
```

### 架构

```
规划 Query × Attention:
  1. Agent-Level Cross-Attention  → 理解其他 agent 的意图
  2. Map-Level Cross-Attention    → 理解道路约束
  3. Ego-Level Self-Attention     → 自车运动推理

         ↓
    解码 → 未来轨迹点: [(x1,y1), ..., (xT,yT)]
         ↓
    MLP → 速度/加速度轮廓
```

### 规划损失

```
L_plan = L2(预测轨迹, 真值轨迹) + Collision Penalty
```

**碰撞惩罚**：如果规划轨迹在某个时间步与任何 Agent 的 BBox 重叠 → 加惩罚。

---

## 五、与 UniAD 的对比

| 维度 | UniAD | VAD |
|------|-------|-----|
| **场景表征** | 密集占据栅格 | 矢量化 token |
| **地图** | 密集分割 | 矢量元素 |
| **规划输入** | 占据 + 轨迹 + 地图 | Scene Tokens |
| **参数量** | 大 | 小 (~1/3) |
| **FPS** | ~1.8 | ~5-10 |
| **可解释性** | 占据可视化 | 矢量可视化 |
| **精度 (L2)** | 较低 | 相近/更好 |

**VAD 的效率来源**：
1. 不做密集占据预测
2. 矢量化 Attention 比密集 Attention 快
3. 不需要 OccFormer 模块

---

## 六、训练细节

### 数据增强

- 场景旋转和平移
- Agent dropout (随机移除某些 agent，学习鲁棒性)

### 损失平衡

```
总损失 = w1 × L_detection + w2 × L_motion + w3 × L_map + w4 × L_plan
         (预训练阶段 )        (端到端阶段)
```

---

## 七、变体与后续

### VADv2

- 更强的 BEV Encoder
- 多帧时序输入
- 闭环评测支持

### GenAD (Generative E2E)

- 将规划建模为生成任务 (VAE-based)
- 输出多模态规划轨迹

---

## 八、关键概念

| 概念 | 含义 |
|------|------|
| **Vectorized Scene** | 矢量化场景，用点/线/多边形的集合表示 |
| **Scene Tokens** | 场景编码 token，包含 agent 和 map 信息 |
| **Planning Transformer** | 轻量规划模型，处理 Scene Tokens |
| **Collision Penalty** | 碰撞惩罚损失 |

---

## 📖 推荐资料

- VAD 论文 (ICCV 2023)
- VAD 开源代码
- MapTR 论文（在线矢量化建图）
- VectorMapNet 论文

---

## 相关笔记

- [[端到端自动驾驶概览]]
- [[UniAD详解]]
- [[BEVFormer详解]]
- [[大模型与自动驾驶]]
