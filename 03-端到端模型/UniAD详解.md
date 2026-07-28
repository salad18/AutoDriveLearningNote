---
tags: [end-to-end, planning, model]
created: "2026-07-21"
updated: "2026-07-28"
---

# UniAD 详解

> **论文**: Planning-oriented Autonomous Driving (CVPR 2023 🏆 **Best Paper Award**)
> **机构**: 上海 AI Lab / 武汉大学 / 商汤科技
> **代码**: [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)

---

## 一、核心思想

UniAD 的哲学：**"Planning is not just a downstream task — it should drive the design of upstream perception modules."**

传统 pipeline: `检测 → 跟踪 → 预测 → 规划`，每个模块独立优化，信息在模块边界被压缩丧失。

UniAD: 所有模块通过 **Query 机制**统一交互，感知模块产生的 Query 特征直接流向规划器。

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

---

## 三、模块详解

### 3.1 TrackFormer — 检测+跟踪统一

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

**追问级的实现细节**:

为什么需要 600 个 Track Queries（远多于实际物体数 ~50）？
- 冗余保证：即使 80% 的 queries 不活跃，仍有 120 个活跃 → 足够覆盖任何场景
- 类似 DETR 的设计哲学：让模型自己学会哪些 query 该活跃
- 消融：300 → NDS -1.2, 900 → NDS +0.1 (边际递减)

**匹配机制** (区别于 DETR 的匈牙利匹配)：

Track Queries 之间的 Self-Attention 使它们**互相排斥**（类似 NMS 但 differentiable）：
- 两个 query 同时 fire 在同一位置 → attention map 会降低其中一个的分数
- 不需要显式 NMS → 训练时用 set prediction loss 隐式学习

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

### 3.3 OccFormer — 占据预测

将未来轨迹预测转化为未来的占据栅格预测。

**机制**: 从 agent trajectories → agent 未来占据位置 → scene-level occupancy:
- 对于已知 agent (通过 Track/Motion 预测的) → 直接投影其未来 bbox 到 BEV grid
- 对于未知区域 → 通过 learned occupancy decoder 补全

**这一模块的设计意图**: 不仅告诉 Planner "agent 在哪"，还告诉它"空间占用将如何变化" → 规划时可以主动避让未来的占据区域。

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

**追问: 为什么不直接端到端训练?**

1. **Loss 平衡问题**: 5 个模块的 loss scale 差异巨大 (track ~5, map ~0.5, occ ~1, plan ~0.01) → 端到端训练会被 track loss 主导
2. **收敛困难**: 早期感知不准 → 给 Motion 错误的输入 → Motion 学错了 → 后续模块全部学歪 (级联错误)
3. **实验证明**: 端到端训练 → NDS 下降 5+ 点

这是 UniAD 论文中不太被强调但**面试中极其重要**的细节。

---

## 五、关键数值 & 消融

### 5.1 整体性能 (nuScenes val)

| 模型 | Detection NDS | Motion minADE↓ | Planning L2↓ | Collision↓ |
|------|:-----------:|:-------------:|:----------:|:-------:|
| UniAD (R101) | 49.3 | 0.71 | 1.03 | 0.12% |
| UniAD (VoV-99) | 54.0 | 0.65 | 0.88 | 0.09% |

注意：UniAD 的 Detection NDS 低于纯 BEVFormer-B (56.9) → 因为多任务训练的干扰 (capacity sharing)。

### 5.2 模块消融

| 去掉的模块 | Δ Planning L2 | 影响 |
|-----------|:------------:|------|
| 去掉 TrackFormer | +0.31 | 大! 不知道其他 agent 在哪 → 规划变保守 |
| 去掉 MapFormer | +0.18 | 中等: 没有车道 → 规划偏差 |
| 去掉 MotionFormer | +0.42 | **最大!** 无法预测未来 → 规划严重不准 |
| 去掉 OccFormer | +0.15 | 有影响但不大 (Motion 已经很准确) |
| 去掉所有交互 (只用 BEV) | +0.89 | 端到端设计的核心价值 |

### 5.3 规划质量 (关键!)

| 指标 | UniAD | 传统 Pipeline |
|------|:-----:|:------------:|
| **Planning L2 (m) ↓** | 0.88 | 1.21 |
| **Collision Rate (%) ↓** | 0.09 | 1.32 |

碰撞率降低 15×！这是 UniAD 最大的卖点 — **联合优化让规划更安全**。

---

## 六、面试官深度追问

### Q: UniAD 的 Track Query 怎么处理新出现的目标和消失的目标？

Track Query 是**固定数量**（600个），通过 confidence score 管理生命周期：
- 出生: confidence > θ_new=0.4 → 激活一个新的 Track Query
- 持续: confidence 保持在 0.4 以上 → 继续活跃
- 死亡: 连续 N=2 帧 confidence < θ_dead=0.2 → 释放该 Query

追问：如果同时出现 10 个新车（如从停车场涌出），600 个 Query 够吗？
→ 够。600 远大于正常场景 (~50 agents)，冗余度 10×。

### Q: 为什么 MotionFormer 预测 6 个模态而不是 3 个或 10 个？

nuScenes 数据集中，交叉路口的 agent 行为最多 6 种 (直行、左转、右转、U-turn、减速、加速)。
消融: 3→6 模态 → minADE 降低 0.15 / 6→10 模态 → minADE 降低 0.03 (边际递减)。

### Q: UniAD 最大的工程挑战？

1. **多阶段训练不稳定**: Stage 依赖 → 如果 Stage 1 训坏了，后面全废 → 需要频繁 checkpoint
2. **显存**: 同时跑 5 个模块 → 8×A100 80GB 单卡 ~65GB 显存 (接近极限)
3. **延迟**: 完整 pipeline ~300ms per frame (3 FPS) → 远不能实时
4. **可复现性**: 官方开源代码和论文结果有 ~2 NDS 差距 (论文更高)

### Q: 既然 2024 年大家说端到端不好落地，UniAD 的价值在哪？

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

---

> 📚 **相关笔记**: [[VAD详解]], [[端到端自动驾驶概览]], [[BEVFormer详解]], [[华为ADS端到端架构]]
> 🎯 **面试题**: [[常见面试题-感知算法]] Q17-18
