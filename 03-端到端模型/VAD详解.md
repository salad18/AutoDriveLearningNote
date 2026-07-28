---
tags: [end-to-end, planning, vectorized]
created: "2026-07-21"
updated: "2026-07-28"
---

# VAD 详解

> **论文**: VAD: Vectorized Scene Representation for Efficient Autonomous Driving (ICCV 2023)
> **机构**: 上海 AI Lab / 华中科技大学
> **代码**: [hustvl/VAD](https://github.com/hustvl/VAD)

---

## 一、核心思想

**矢量化 = 只存有用的信息。** 一个 BEV 占据栅格 640K 体素中 90% 是"空"或"路面"，VAD 直接将这些信息压缩为稀疏的矢量表示（agent 轨迹、车道线坐标），将 Planner 的计算量降低一个数量级。

```
UniAD:  Image → Dense BEV (640K cells) → Dense Occupancy → Planner  [3 FPS]
VAD:    Image → BEV Features → Vectorized Agents + Lanes → Planner  [10+ FPS]
```

**追问: 为什么矢量化就够了？驾驶决策需要哪些信息？**

驾驶决策的最小信息集合：
1. 周围 agent 的位置和运动 → 矢量化轨迹
2. 车道拓扑 → 矢量化的 lane graph
3. 交通灯状态 → 分类标签
4. 自车状态 → 速度/加速度/朝向

这些全部是**稀疏的**（数量 O(10-100)，而非 640K）→ 矢量化天然匹配。

---

## 二、架构参数

| 参数 | UniAD | VAD | 改进 |
|------|-------|-----|------|
| Backbone | R101/VoV-99 | **R50** | 更轻 |
| Agent 表示 | Track Query (dense) | **Vectorized polyline** | 稀疏 |
| Map 表示 | Map Query (dense) | **Lane graph** | 结构化 |
| Planner 输入 | Dense features | **Sparse vectors** | 高效 |
| FPS | ~3 | **~10** | 3× |
| Planning L2 ↓ | 0.88 | **0.80** | ✓ |
| Collision ↓ | 0.09% | **0.07%** | ✓ |

---

## 三、模块详解

### 3.1 Vectorized Motion (Agent 矢量化)

**输出格式**: 每个 agent 的轨迹 = 一系列 (x, y, t) 点，而非 dense heatmap。

```python
# 传统 (UniAD):
# agent_state = dense_heatmap[200, 200] + bbox_regression
# 一个 agent 需要: heatmap peak detection + bbox decode

# VAD:
# agent_traj = [(x_t1, y_t1), (x_t2, y_t2), ..., (x_t6, y_t6)]
# 直接输出 6 个未来时间步的 (x,y) 坐标

# 优势:
# 1. 不需要 dense prediction → 轻量
# 2. 直接消费于 planner → 不需要后处理
# 3. 可解释: 我能看到模型预测的轨迹
```

**Agent-Agent 交互**: 用 Self-Attention between agent vectors → 建模交互（如变道、让行）→ 关键创新！

```python
# agent vectors 之间的 Self-Attention
agent_feats = [feat_1, feat_2, ..., feat_N]  # N agents
agent_feats = SelfAttention(agent_feats)
# → 每个 agent 现在知道了其他 agent 的意图
```

### 3.2 Vectorized Map (车道矢量化)

类似 MapTR，将车道线表示为 polyline（折线）：

```
车道 Centerline: [(x₁,y₁), (x₂,y₂), ..., (x₂₀,y₂₀)]  (20 points sampled)
车道 Divider:    [(x₁,y₁), (x₂,y₂), ...]
人行横道:        polygon vertices [(x₁,y₁), (x₂,y₂), (x₃,y₃), (x₄,y₄)]
```

**为什么用 polyline 而不像 HD Map 用参数化曲线？**
- Polyline 足够灵活（任何形状）
- 直接输出有序点集 → 不需要拟合
- 和 Transformer decoder 天然匹配（each point = a query）

### 3.3 Planner — 矢量空间的显式规划

这是 VAD 最亮眼的创新：**在矢量空间中做显式碰撞检测和规划优化**。

```python
# 传统方法 (UniAD): 用 NN 黑盒预测规划轨迹
# VAD: 用矢量信息做显式 optimization

def vad_planner(ego_state, agent_trajs, lane_graph, nav_cmd):
    # 1. 采样候选轨迹 (根据车道 + 导航)
    candidates = sample_trajectories(lane_graph, nav_cmd)  
    # → K 条候选轨迹 (如 K=32)
    
    # 2. 对每条候选轨迹评分
    scores = []
    for traj in candidates:
        # 安全分: 与 agent 轨迹的最近距离
        safety = min_distance(traj, agent_trajs)
        # 舒适分: 加速度 / 曲率
        comfort = smoothness(traj)
        # 进度分: 向目标前进的距离
        progress = toward_goal(traj, nav_cmd)
        
        score = w1*safety + w2*comfort + w3*progress
    
    # 3. 选最高分的轨迹
    best_traj = candidates[argmax(scores)]
    return best_traj
```

**关键设计**: 候选轨迹采样 + 矢量化评分 → **可解释、可调试**。如果车跑偏了，可以检查是 safety 分太低还是 progress 分太低。

**追问: 为什么不用 RL 直接输出规划？**

因为 RL 在安全关键场景中不可解释。VAD 的显式优化方法：
- 每个评分项 (safety/comfort/progress) 可独立调参
- 可以人工注入安全约束 (min_distance < 1m → force 0 score)
- 出了问题可以回溯: 是采样不够密还是评分函数有问题

---

## 四、训练细节

### 训练配置

| 项目 | 值 |
|------|-----|
| Backbone | ResNet-50 (ImageNet pretrain) |
| BEV Encoder | 简化版 BEVFormer (3 layers vs 6) |
| Epochs | 60 (nuScenes) |
| Batch Size | 4 (8 GPUs × 0.5 per GPU, gradient accumulation) |
| LR | 2×10⁻⁴, Cosine Annealing |
| 训练时间 | ~36h (8×A100) |

### 损失函数

```
L_total = L_det + L_map + L_motion + L_plan

L_det (agent检测):  Focal Loss + L1 (类似 DETR)
L_map (车道检测):   L1 for polyline points
L_motion (轨迹预测): L2 for agent trajectory points
L_plan (规划):      L2 imitation (vs human driver traj)
```

---

## 五、面试官深度追问

### Q: VAD 的矢量输出如果 agent 数量变化很大怎么办？

VAD 的 agent query 也是固定数量（如 300），类似 DETR/DETR3D 的设计。多余 query 输出 "no object" 类别。Agent 数量变化在 ±30 范围内 → 300 queries 有足够冗余。

### Q: 矢量化丢失了哪些信息？有什么代价？

丢失的信息：
1. **稠密空间信息**: 占据栅格中有 "路面" vs "人行道" 的精细信息 → 矢量只存车道线
2. **未知障碍物**: 如果有个 agent 检测漏了 → 占据栅格还能靠几何发现障碍 → 矢量没有
3. **Cost**: 矢量化的车道需要额外训练 Map Head → 增加了模型复杂度

这就是为什么最新的华为 ADS 3.0 同时输出 GOD (稠密占据) + 矢量检测 — **稠密做安全兜底，稀疏做高效规划**。

### Q: 采样 32 条候选轨迹够吗？为什么不用 optimization-based planner？

采样-based 的优缺点：

优点: 简单、并行、可解释
缺点: 32 条可能不够 → 最优轨迹可能在采样空间之外

实际工程中，32 条配合 lane graph 引导覆盖了 >95% 的正常驾驶场景。对于极端场景（如紧急避让）→ 需要增加采样密度或切换到 optimization-based。

目前趋势 (2024): 采样 + optimization 混合 — 采样做粗搜索 → optimization 做精调。

### Q: VAD 能处理无保护左转吗？（面试高频！）

VAD 可以，但依赖于 agent motion prediction 的质量。无保护左转需要：
1. 准确预测对向车辆的未来轨迹（Motion Head）
2. 候选轨迹 scoring 中高权重 safety → 只在安全时执行左转
3. 但**如果 agent motion 预测错误**（对向车突然加速），VAD 无法实时适应 → 需要闭环反馈

这是所有端到端模型的共同弱点 — 对罕见/对抗场景的泛化有限。

---

## 六、VAD 与 UniAD 的核心差异总结

| | UniAD | VAD |
|---|---|---|
| **设计哲学** | 感知特征服务规划 | 矢量表示服务规划 |
| **规划方式** | 隐式 (NN 黑盒) | **显式** (candidate scoring) |
| **可解释性** | 低 | **高** |
| **计算量** | 大 (dense throughout) | 小 (sparse after encoding) |
| **安全验证** | 难 | 易 (可审计 scoring) |
| **落地前景** | 研究 | **量产可行** |

**关键洞察**: VAD 证明了 "稀疏表示 + 显式规划" 不仅在效率上优于 UniAD，在精度和安全性上也有优势 — 这是 2023-2024 年端到端自动驾驶最关键的范式转换。

---

> 📚 **相关笔记**: [[UniAD详解]], [[端到端自动驾驶概览]], [[华为ADS端到端架构]], [[BEVFormer详解]]
> 🎯 **面试题**: [[常见面试题-感知算法]] Q17-18
