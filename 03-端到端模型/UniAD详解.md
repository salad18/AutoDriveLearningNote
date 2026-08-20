---
tags: [end-to-end, planning, model]
created: "2026-07-21"
updated: "2026-08-21"
---

# UniAD 详解

> **论文**: Planning-oriented Autonomous Driving (CVPR 2023 🏆 **Best Paper Award**)
> **机构**: 上海 AI Lab / 武汉大学 / 商汤科技
> **代码**: [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | UniAD 的核心哲学（Planning-Oriented）、四大模块（TrackFormer / MotionFormer / OccFormer / Planner）的机制、分阶段训练策略、消融实验、面试追问 |
| **需要的前置知识** | [[端到端自动驾驶概览]]（端到端概念、开环/闭环）、[[BEVFormer详解]]（Query 机制）、[[Transformer架构详解]]（Attention） |
| **学完之后你能** | ① 画出 UniAD 完整数据流（检测→跟踪→预测→占据→规划）；② 解释"为什么 Track Query 不需要卡尔曼滤波"；③ 说清"为什么不能直接端到端训练"；④ 回答"UniAD 的价值在哪"这类面试题 |
| **预计阅读时间** | 90-120 分钟 |

> [!tip] 本篇是你读懂的第一篇端到端论文
> UniAD 是"感知-规划一体化"的代表作（[[端到端自动驾驶概览]] 流派 3）。读之前请确保两件事：① 记得"端到端的切分点"概念（UniAD 的切分点在**规划输出**，控制还是规则）；② 记得 BEVFormer 的 Query 机制（UniAD 把它发扬光大了）。

---

## 〇、大白话总览（先读这个！）

### UniAD 在做什么？

传统方案：**检测器 → 跟踪器 → 预测器 → 规划器**，四个独立模块，每个模块的输出（框、ID、轨迹）是给下一个模块的"手工接口"。

UniAD：**一张网里同时做六件事**——检测、跟踪、地图（车道线）、预测、占据、规划，全部通过 **Query（查询向量）** 串联：

```
图像 → BEV Encoder
        ↓
   TrackFormer（检测+跟踪）→ 目标查询
        ↓
   MapFormer（车道线）→ 地图查询
        ↓
   MotionFormer（轨迹预测）→ 每个目标 6 种未来
        ↓
   OccFormer（未来占据）→ 3D 占据栅格
        ↓
   Planner（规划）→ 自车轨迹
```

### 核心思想（一句话）

**"规划不是下游任务，而应该反过来驱动上游感知模块的设计"** —— 感知不是为感知而感知，而是为了**让规划开得安全**。所以 UniAD 里每个感知模块都在为规划服务：检测是为了知道"谁在哪"，预测是为了知道"他们要去哪"，占据是为了知道"哪些空间会被占用"。

### 最亮眼的成绩

**碰撞率比传统 pipeline 降低 15 倍**（1.32% → 0.09%）——联合优化让规划更安全，这是它拿 Best Paper 的核心卖点。

---

## 一、核心思想（展开版）

UniAD 的哲学：**"Planning is not just a downstream task — it should drive the design of upstream perception modules."**

传统 pipeline: `检测 → 跟踪 → 预测 → 规划`，每个模块独立优化，信息在模块边界被压缩丧失。

UniAD: 所有模块通过 **Query 机制**统一交互，感知模块产生的 Query 特征直接流向规划器。

> [!note] "Planning-Oriented"到底是什么意思？
> 传统方案里，规划是"最后一站"——前面模块好不好，只看各自指标（检测看 mAP、预测看 minADE），**没人关心"这些模块对规划有没有用"**。
> UniAD 反过来想：**如果某个感知信息对规划没用，为什么要算它？如果规划需要某种信息（比如"未来占据空间"），感知就该提供它。** 这就是"以规划为导向设计感知"——四个模块的消融实验（见第五节）证明了每个模块都对规划有用，缺一个规划就差一截。

---

## 二、架构参数速查

| 参数 | 值 |
|------|-----|
| BEV 分辨率 | $200 \times 200 \times 256$ |
| BEV 范围 | $[-51.2m, 51.2m]^2$ |
| Track Queries 数量 | $600$ (远超场景中实际目标数) |
| Map Queries 数量 | $100 \times 3$ 类 (divider, ped_crossing, boundary) |
| Motion Query 数量 | $600$ (与 Track Query 一一对应) |
| Motion 模态数 | $6$ (每个 agent 预测 6 种可能未来) |
| 预测时域 | $3s$ (future) |
| OccFormer 体素 | $200 \times 200 \times 16$ |
| Backbone | ResNet-101 / VoVNet-99 |
| 训练 Epochs | 20 (2-stage: 6 track + 6 map + 8 full) |

> [!note] 一组数字建立规模感
> **600 个 Track Query** 应对场景中通常只有约 50 个目标——10 倍冗余，为什么？后面 TrackFormer 部分细讲。**6 种模态**——每个目标预测 6 条可能的未来轨迹（直行/左转/右转/……），因为未来不确定，模型要"多条腿走路"。

---

## 三、模块详解

### 3.1 TrackFormer — 检测 + 跟踪统一

**核心创新**: Track Queries 在时序中**持续存在**，隐式携带 identity 信息。

```python
# Track Query 的生命周期:
# 1. 初始化: 随机初始化 600 个 Track Queries + 600 个 Detect Queries
# 2. 每帧 forward:
#    - 上一帧的 Track Queries (带着历史信息) + 新 Detect Queries → Self-Attention
#    - Attention 自动学习"匹配": 哪些 Track Query 对应哪个 Detect Query
#    - 输出: Track Query → bbox + velocity + confidence
# 3. 生命周期管理:
#    - confidence > θ_new  → 新的 Track Query 出生
#    - confidence < θ_dead  → 该 Track Query 死亡 (连续 T 帧低分数)

# 关键: 不需要 IoU matching! 不需要 Kalman Filter!
# Track Query 本身的 self-attention 就完成了"匹配"
```

> [!note] 为什么"不需要卡尔曼滤波"？（这是理解 UniAD 的关键）
> 传统跟踪（[[计算机视觉基础]] 第五节）需要"卡尔曼滤波预测位置 + 匈牙利算法匹配检测框"——因为检测是"每帧独立的"，要手工把两帧的框关联起来。
> UniAD 的 Track Query **跨帧持续存在**：这个 query 上一帧是"那辆车"，这一帧还是"那辆车"（它带着历史信息参与 attention）。**身份（identity）被 Query 本身携带，不需要匹配算法**——检测和跟踪在同一个机制里同时完成。这就是"联合"（Joint）二字的含义。

**追问级的实现细节**：

为什么需要 600 个 Track Queries（远多于实际物体数 ~50）？
- 冗余保证：即使 80% 的 queries 不活跃，仍有 120 个活跃 → 足够覆盖任何场景
- 类似 DETR 的设计哲学：让模型自己学会哪些 query 该活跃
- 消融：300 → NDS -1.2, 900 → NDS +0.1 (边际递减)

**匹配机制**（区别于 DETR 的匈牙利匹配）：

Track Queries 之间的 Self-Attention 使它们**互相排斥**（类似 NMS 但 differentiable）：
- 两个 query 同时 fire 在同一位置 → attention map 会降低其中一个的分数
- 不需要显式 NMS → 训练时用 set prediction loss 隐式学习

> [!note] "互相排斥"的直觉
> 600 个 query 都想抢"那辆车"，怎么办？Self-Attention 让它们互相"商量"：**同一个位置只需要一个 query 站出来**（其他自动降低置信度）。这像可微分的 NMS——传统 NMS 是后处理的硬规则，这里是网络内部学出来的软机制。

### 3.2 MotionFormer — 多模态轨迹预测

**公式**: 对每个 agent $i$，预测 $M=6$ 种可能未来：

$$\{(\hat{p}_{i,m}^{t+1}, \hat{p}_{i,m}^{t+2}, ..., \hat{p}_{i,m}^{t+T})\}_{m=1}^{M}$$

其中 $\hat{p}_{i,m}^{t+\tau} \in \mathbb{R}^2$ 是 $t+\tau$ 时刻的预测位置。

**核心设计**:
1. Motion Query = Track Query + 场景上下文 (Map Query + 自车状态) → Self-Attention
2. 每个 Motion Query 编码 agent 的意图 (左转/右转/直行)
3. 多模态输出通过 **mode attention** (softmax over 6 modes) 得到
4. Loss: 只监督预测轨迹和 GT 最近的 mode (best-of-K loss)

```python
# 简化 MotionFormer 核心
motion_query = track_query + self_attention(track_query, map_query, ego_state)
# [B, N_track, 256]

modes = motion_head(motion_query)  # [B, N_track, M=6, T=6, 2]
# 6 modes × 6 timesteps × (x, y)

# 只训练最接近 GT 的 mode:
closest_mode = argmin(distance(modes, gt_traj), dim=mode_dim)
loss = L1_loss(modes[closest_mode], gt_traj)
```

> [!note] 为什么预测"6 种未来"而不是"1 种"？
> 未来不确定：路口那辆车可能直行、可能左转——**预测单一轨迹 = 逼模型"猜一个"，猜错了整个规划就错**。预测 6 种（带概率）让模型表达"不确定性"：规划时对每种可能都要安全（保守策略）。
> **best-of-K loss**：训练时只惩罚"离真值最近的那个 mode"——因为模型只需要"至少有一条轨迹猜对"，剩下的 mode 自由发挥表达其他可能性。这是多模态预测的标准训练技巧。

### 3.3 OccFormer — 占据预测

将未来轨迹预测转化为未来的占据栅格预测。

**机制**: 从 agent trajectories → agent 未来占据位置 → scene-level occupancy:
- 对于已知 agent (通过 Track/Motion 预测的) → 直接投影其未来 bbox 到 BEV grid
- 对于未知区域 → 通过 learned occupancy decoder 补全

**这一模块的设计意图**: 不仅告诉 Planner "agent 在哪"，还告诉它"空间占用将如何变化" → 规划时可以主动避让未来的占据区域。

> [!note] OccFormer 和 [[占据网络与GOD]] 的区别
> 占据网络预测**当前**的 3D 占据；OccFormer 预测**未来**的 2D 占据（未来 3 秒每时刻的 BEV 占据栅格）。**从"现在哪里有东西"到"将来哪里会有东西"**——这是"预测"和"感知"的交叉，也是规划真正需要的信息（规划要避让的是**未来**的位置）。

### 3.4 Planner — 规划器

**输入**:
- MotionFormer 输出: 所有 agent 的未来轨迹
- OccFormer 输出: 未来占据栅格
- 导航信息: 高级指令 (直行/左转/右转)

**规划损失** (这是 UniAD 最核心的贡献！):

$$L_{plan} = \lambda_1 L_{imitation} + \lambda_2 L_{safety} + \lambda_3 L_{comfort}$$

其中：
- $L_{imitation}$: 预测轨迹与人类驾驶员轨迹的 L2 距离 (模仿学习)
- $L_{safety}$: 碰撞惩罚 — 规划的轨迹不能穿过 OccFormer 预测的占据区域
- $L_{comfort}$: 加速度/jerk 正则化 — 平滑的驾驶

```python
# 碰撞惩罚的实现:
def collision_loss(planned_traj, occ_grid):
    # planned_traj: [B, T, 2] (自车未来位置)
    # occ_grid: [B, T, 200, 200] (未来占据概率)

    # 将轨迹点投影到 BEV grid
    traj_indices = world_to_grid(planned_traj)  # [B, T, 2]

    # 采样该位置的占据概率
    occ_at_traj = F.grid_sample(occ_grid, traj_indices)
    # [B, T] — 轨迹点处的占据概率

    # 惩罚高占据概率的轨迹 (越大 = 越危险)
    return occ_at_traj.mean()
```

> [!note] 三项损失各管一件事（这也是"以规划为导向"的体现）
> - **L_imitation（模仿）**：向人类学习——"开得像人一样"。
> - **L_safety（安全）**：**轨迹穿过的位置如果未来被占据，就罚**——直接和 OccFormer 的预测耦合，把"预测"和"规划"绑在一起优化。这是 UniAD 区别于"预测完就扔给规划器"的传统方案的核心差异。
> - **L_comfort（舒适）**：加速度、jerk（加加速度）正则化——"别急刹急转"。
>
> **三项缺一不可**：只模仿 → 学人类但不安全；只安全 → 龟速贴边不敢走；只舒适 → 平滑但乱开。

---

## 四、训练策略 (CRITICAL for interview!)

**不是一次性端到端！UniAD 使用分阶段训练**:

```
Stage 1 (6 epochs): 训练 BEV Encoder + TrackFormer + MapFormer
  → 冻结权重

Stage 2 (6 epochs): 训练 MotionFormer
  → 冻结前两个阶段权重

Stage 3 (8 epochs): 训练 OccFormer + Planner
  → 冻结前三个阶段权重

总计: 20 epochs across 8×A100 → ~4 days
```

> [!question] 追问: 为什么不直接端到端训练？（面试高频！）
> 1. **Loss 平衡问题**: 5 个模块的 loss scale 差异巨大 (track ~5, map ~0.5, occ ~1, plan ~0.01) → 端到端训练会被 track loss 主导
> 2. **收敛困难**: 早期感知不准 → 给 Motion 错误的输入 → Motion 学错了 → 后续模块全部学歪 (级联错误)
> 3. **实验证明**: 端到端训练 → NDS 下降 5+ 点

> [!warning] 这是 UniAD 论文中不太被强调但**面试中极其重要**的细节
> **"端到端"不等于"端到端训练"！** UniAD 的架构是端到端的（一条链路、梯度可以贯穿），但训练是分阶段的（Stage 1 冻结 → Stage 2 → Stage 3）。
> 为什么？**因为多任务联合训练的收敛太难**：5 个任务的 loss 尺度差 500 倍，感知还没学好的时候预测/规划就在学错误输入。分阶段 = **"先把地基打牢，再往上盖楼"**——每个阶段只学一层，用前一层已冻结的稳定特征。
> 记住这个区分，面试时能说出"UniAD 是架构端到端、训练分阶段"直接加分。

---

## 五、关键数值 & 消融

### 5.1 整体性能 (nuScenes val)

| 模型 | Detection NDS | Motion minADE↓ | Planning L2↓ | Collision↓ |
|------|:-----------:|:-------------:|:----------:|:-------:|
| UniAD (R101) | 49.3 | 0.71 | 1.03 | 0.12% |
| UniAD (VoV-99) | 54.0 | 0.65 | 0.88 | 0.09% |

注意：UniAD 的 Detection NDS 低于纯 BEVFormer-B (56.9) → 因为多任务训练的干扰 (capacity sharing)。

> [!note] 一个反直觉但重要的观察
> UniAD 的**检测精度反而低于单任务 BEVFormer**（49.3 vs 56.9）——因为它把网络容量分给了 6 个任务（capacity sharing）。
> **这恰恰证明了它的价值逻辑**：检测低了 7 个点没关系，**规划碰撞率降了 15 倍**——以规划为导向，指标取舍是"牺牲一点感知精度，换取巨大的安全提升"。

### 5.2 模块消融（必考！）

| 去掉的模块 | Δ Planning L2 | 影响 |
|-----------|:------------:|------|
| 去掉 TrackFormer | +0.31 | 大! 不知道其他 agent 在哪 → 规划变保守 |
| 去掉 MapFormer | +0.18 | 中等: 没有车道 → 规划偏差 |
| 去掉 MotionFormer | +0.42 | **最大!** 无法预测未来 → 规划严重不准 |
| 去掉 OccFormer | +0.15 | 有影响但不大 (Motion 已经很准确) |
| 去掉所有交互 (只用 BEV) | +0.89 | 端到端设计的核心价值 |

> [!warning] 读这张表的三个要点
> ① **MotionFormer 影响最大**（+0.42）——预测未来是规划最重要的输入；② **"去掉所有交互"影响最大**（+0.89）——Query 之间的交互机制是端到端设计的灵魂，比任何单个模块都重要；③ 每个模块对规划都有正贡献——**"以规划为导向"不是口号，是消融验证过的**。

### 5.3 规划质量 (关键!)

| 指标 | UniAD | 传统 Pipeline |
|------|:-----:|:------------:|
| **Planning L2 (m) ↓** | 0.88 | 1.21 |
| **Collision Rate (%) ↓** | 0.09 | 1.32 |

碰撞率降低 15×！这是 UniAD 最大的卖点 — **联合优化让规划更安全**。

---

## 六、面试官深度追问

### Q1: UniAD 的 Track Query 怎么处理新出现的目标和消失的目标？

Track Query 是**固定数量**（600个），通过 confidence score 管理生命周期：
- 出生: confidence > θ_new=0.4 → 激活一个新的 Track Query
- 持续: confidence 保持在 0.4 以上 → 继续活跃
- 死亡: 连续 N=2 帧 confidence < θ_dead=0.2 → 释放该 Query

追问：如果同时出现 10 个新车（如从停车场涌出），600 个 Query 够吗？
→ 够。600 远大于正常场景 (~50 agents)，冗余度 10×。

### Q2: 为什么 MotionFormer 预测 6 个模态而不是 3 个或 10 个？

nuScenes 数据集中，交叉路口的 agent 行为最多 6 种 (直行、左转、右转、U-turn、减速、加速)。
消融: 3→6 模态 → minADE 降低 0.15 / 6→10 模态 → minADE 降低 0.03 (边际递减)。

### Q3: UniAD 最大的工程挑战？

1. **多阶段训练不稳定**: Stage 依赖 → 如果 Stage 1 训坏了，后面全废 → 需要频繁 checkpoint
2. **显存**: 同时跑 5 个模块 → 8×A100 80GB 单卡 ~65GB 显存 (接近极限)
3. **延迟**: 完整 pipeline ~300ms per frame (3 FPS) → 远不能实时
4. **可复现性**: 官方开源代码和论文结果有 ~2 NDS 差距 (论文更高)

### Q4: 既然 2024 年大家说端到端不好落地，UniAD 的价值在哪？

价值不在落地，在**范式定义**：
- 是第一个系统性地用 Query 机制统一检测+跟踪+预测+规划的框架
- 证明了 "Planning-oriented" 设计的有效性 (碰撞率降低 15×)
- 启发了后续更实用的工作: VAD（矢量化加速）、GenAD（生成式规划）
- 类似 ResNet 在 CV 中的地位 — 你可能不用 ResNet 但你必须理解它

---

## 七、与 VAD/GenAD 的对比

| | UniAD | VAD | GenAD |
|---|---|---|---|
| **表征** | Dense BEV + Query | **Vectorized** | Query + Generative |
| **规划方式** | 模仿学习 + 安全约束 | 矢量空间显式规划 | 生成式 (diffusion) |
| **速度** | ~3 FPS | ~10 FPS | ~5 FPS |
| **可解释性** | 低 | 高 (矢量可解释) | 低 |
| **落地前景** | 研究 | 量产的中间步骤 | 前沿探索 |
| **年份** | CVPR 2023 | ICCV 2023 | ECCV 2024 |

> [!note] 为什么 VAD 比 UniAD 快 3 倍？
> UniAD 用**稠密 BEV 特征 + 600 个 query**（信息多但算力大）；VAD 用**矢量化表征**（车道线/轨迹变成少量向量，不是稠密网格）——"只保留有用的线，不保留整张图"，计算量骤降（[[VAD详解]] 细讲）。**"表征的稀疏程度"决定了端到端框架的速度上限**，这是 UniAD → VAD 演进的核心线索。

---

## 八、常见误区速查

> [!warning] 新手最常踩的坑
> 1. **以为 UniAD 是"图像→方向盘"**——不是！它输出的是**自车轨迹（规划）**，控制还是规则模块。端到端的切分点在规划。
> 2. **以为 Track Query 需要卡尔曼滤波/匈牙利匹配**——不需要！身份由 Query 跨帧携带，Self-Attention 自动完成匹配（这是"联合检测跟踪"的核心）。
> 3. **以为 UniAD 是"端到端训练"**——架构端到端，训练**分三阶段**（track/map → motion → occ/plan），直接端到端训练会因 loss 尺度失衡和级联错误崩掉（NDS -5）。
> 4. **只看"检测 NDS 不如 BEVFormer"就否定它**——它的目标是规划安全（碰撞率 -15×），感知精度适度让步是"以规划为导向"的取舍。
> 5. **忽略"去掉所有交互"的消融**——Query 交互机制（+0.89）比任何单个模块都重要，这是端到端设计的灵魂。

---

## ✅ 检验自己（自测题）

> [!question] Q1：用一句话解释"Planning-Oriented"（以规划为导向）的设计哲学，并举一个具体体现。
> 提示：感知模块为什么存在？

> [!success]- 参考答案
> "规划不是下游任务，而应驱动上游感知设计"——感知模块存在的意义是让规划开得更安全，不是单独刷感知指标。具体体现：OccFormer 预测**未来占据**（而不是当前占据），因为规划要避让的是未来位置；规划损失直接和 OccFormer 输出耦合（轨迹穿过未来占据区域就罚），把预测和规划绑在一起优化。

> [!question] Q2：Track Query 为什么能同时完成"检测 + 跟踪"？它和"卡尔曼滤波 + 匈牙利匹配"的本质区别是什么？
> 提示：身份信息从哪来？

> [!success]- 参考答案
> Track Query 跨帧持续存在，身份（identity）由 Query 自身携带——上一帧的 query 带着历史信息参与这一帧的 attention，天然知道"我还是那辆车"。传统方案检测每帧独立，需要卡尔曼滤波预测位置 + 匈牙利算法匹配两帧框来"猜"身份；UniAD 不需要匹配，Self-Attention 自动完成关联。这是"联合"（Joint）的本质。

> [!question] Q3：为什么 600 个 Track Query？为什么不是 50 个（场景目标数）？
> 提示：冗余和不确定性。

> [!success]- 参考答案
> ① 冗余保证：场景目标数不确定（高峰 100+ 个），600 个即使 80% 不活跃也有 120 个可用；② DETR 设计哲学：让模型自己学哪些 query 该活跃，而不是手工固定数量；③ 消融显示 300 个会掉 1.2 NDS，900 个只涨 0.1（边际递减），600 是效率与精度的平衡点。

> [!question] Q4：规划损失的三项分别管什么？为什么"模仿 + 安全 + 舒适"缺一不可？
> 提示：各管一个维度。

> [!success]- 参考答案
> L_imitation（模仿）：预测轨迹贴近人类驾驶（学人类行为）；L_safety（安全）：轨迹穿过 OccFormer 预测的未来占据区域就罚（防碰撞）；L_comfort（舒适）：加速度/jerk 正则化（驾驶平顺）。缺一不可：只模仿可能不安全（人类也会犯错）、只安全会龟速保守不敢走、只舒适会平滑但乱开。三者权衡才能"像人一样安全舒适地开"。

> [!question] Q5：UniAD 为什么不能直接端到端训练？分阶段训练解决的是什么问题？
> 提示：loss 尺度 + 级联错误。

> [!success]- 参考答案
> ① Loss 平衡问题：5 个模块 loss 尺度差巨大（track ~5 vs plan ~0.01），端到端训练被 track loss 主导，规划几乎学不到；② 收敛困难：早期感知不准 → 给 Motion 错误输入 → Motion 学歪 → 后续模块级联错误。分阶段（track/map → motion → occ/plan，前阶段冻结）先打牢地基再盖楼，用稳定特征训练后续模块。实验证明直接端到端训练 NDS 掉 5+ 点。

> [!question] Q6：UniAD 的检测 NDS 低于单任务 BEVFormer，为什么还说它更好？
> 提示：指标取舍，规划安全。

> [!success]- 参考答案
> UniAD 把网络容量分配给 6 个任务（capacity sharing），单任务检测精度自然下降（49.3 vs 56.9）。但它的目标不是感知指标，而是规划安全——碰撞率 1.32% → 0.09%（降低 15 倍）。"以规划为导向"的取舍逻辑：感知精度适度让步，换取巨大的安全提升。这也说明端到端评测要看规划指标，不能只看感知 mAP。

---

## 🛠 动手练习

### 练习 1：画出 UniAD 完整数据流（30 分钟）

用 Excalidraw 或纸笔画出：
1. 六个任务模块（检测/跟踪/地图/预测/占据/规划）的连接关系。
2. 标注每个模块的输入（哪些 Query/特征）和输出。
3. 用红笔标出"端到端切分点"（UniAD 输出到哪为止）。

> [!tip] 画完自检
> ① Track Query 流向 MotionFormer 了吗？② OccFormer 的输出流向 Planner 了吗？③ Planner 的输入有几路？（Motion + Occ + 导航指令）——三问都对，数据流就通了。

### 练习 2：读 UniAD 论文的消融部分（60-90 分钟）

打开 UniAD 论文（arXiv: 2212.11294），读 Table 4（模块消融）和规划评测部分，回答：
1. 论文怎么证明"每个模块对规划都有贡献"？（对照本文 5.2 节）
2. 它用的是什么评测协议（开环还是闭环）？指标是什么？
3. "去掉所有交互"为什么影响最大？用你自己的话解释。

> [!tip] 这是"端到端论文批判性阅读"的完整练习
> 先看评测（开环/闭环）、再看消融（哪些模块真的有用）——这个习惯会贯穿你以后读的所有端到端论文。

### 练习 3：跑通 UniAD 推理（可选，1-2 天）

按官方仓库（GitHub: OpenDriveLab/UniAD）在 nuScenes mini 上跑通推理：
1. 配置环境（官方 Docker 推荐，mmdet3d 版本敏感）。
2. 跑一次推理，可视化：检测框、跟踪轨迹、预测的 6 种模态、规划轨迹。
3. 观察"6 种模态"里不同 mode 的差异（直行/转弯的轨迹分支）。

> [!warning] 环境提示
> UniAD 代码对 mmdet3d/mmcv/PyTorch 版本要求极严，环境配置 2-3 小时起步是正常的。**跑不通先看 [[训练排错实战手册]]，把报错记进 [[2026-08-20]]（今日日记）**——环境调试本身就是学习。

---

## ➡️ 下一步学什么

按知识库学习路径，读完本篇你应该接着：

1. **[[VAD详解]]** —— 对照学习：矢量化表征如何让端到端快 3 倍（UniAD → VAD 的演进主线）。
2. **[[华为ADS端到端架构]]** —— 量产视角：GOD + PDP 分体式端到端 + CAS 3.0 安全兜底（UniAD 的研究思路如何在量产中变形）。
3. **[[大模型与自动驾驶]]** —— 进阶：DriveVLM 等把语言推理带进驾驶。
4. **[[世界模型]]** —— 进阶：OccWorld 等预测未来世界的生成式方法。

> 💡 恭喜！你走完了新手路径 9 站（基础理论 5 篇 + BEV 4 篇 + 端到端 2 篇的骨架）。接下来建议：① 回 [[学习计划]] 打勾；② 读 [[VAD详解]] 完成端到端对照；③ 开始 [[小车项目总览]] 的动手项目，把知识变成动手能力。

---

## 相关笔记

- [[端到端自动驾驶概览]] — 端到端概念基础
- [[VAD详解]] — 矢量化端到端对照
- [[BEVFormer详解]] — Query 机制基础
- [[华为ADS端到端架构]] — 量产端到端实践
- [[大模型与自动驾驶]] — LLM/VLM 驱动驾驶
- [[世界模型]] — 未来世界预测
- [[常见面试题-感知算法]] — 面试实战
