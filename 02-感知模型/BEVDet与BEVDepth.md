---
tags: [BEV, detection, LSS]
created: "2026-07-21"
updated: "2026-08-21"
---

# BEVDet 与 BEVDepth

> 一句话导读：BEVDet 是"显式深度派"（LSS 范式）的工业级标杆——**先让网络猜每个像素离多远，再把特征"抬"到 3D、拍平到鸟瞰图**；BEVDepth 针对它最大的弱点（深度猜不准）给出解药：**用 LiDAR 的真值教相机猜深度**。这篇把两条主线讲透：LSS 怎么工作、为什么深度监督这么关键。

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | LSS 三步骤（Lift/Splat/Shoot）的完整机制、BEVDet 架构、BEVDet 为什么比 BEVFormer 快、BEVDepth 深度监督原理、BEVPoolv2 工程优化、BEVDet4D 时序扩展 |
| **需要的前置知识** | [[BEV感知全景]]（BEV 整体认知）、[[3D视觉与投影几何]]（投影、坐标系变换）、[[计算机视觉基础]]（检测头） |
| **学完之后你能** | ① 用自己的话讲清楚 LSS 三步；② 说出 BEVDepth 比 BEVDet 强在哪、为什么；③ 解释 BEVDet 在部署上比 BEVFormer 的工程优势；④ 回答"纯视觉没有 LiDAR 怎么监督深度"这类面试追问 |
| **预计阅读时间** | 60-90 分钟 |

> [!tip] 本篇和 [[BEVFormer详解]] 是对照组
> 同样做 BEV 感知，LSS 派"显式量距离"，BEVFormer 派"隐式学距离"。**推荐先读本篇、紧接读 BEVFormer**，两篇对照，两条路线同时吃透。

---

## 一、LSS 范式回顾——"Lift-Splat-Shoot"到底是什么

LSS 来自论文 *Lift, Splat, Shoot*（Philion & Fidler, ECCV 2020），三个词对应三个步骤：

```
Lift:   像素特征 × 深度分布 (D bins) → 3D 视锥体特征
        每个像素 (u,v): f_c(u,v) × d(u,v) → [C, D] 的锥体特征
Splat:  锥体特征 → 利用相机内外参变换到 ego 坐标系 → Sum/Voxel Pooling → BEV
Shoot:  BEV 特征 → BEV Backbone (ResNet) → Detection Head (CenterPoint)
```

### Step 1: Lift（抬升）——给每个像素"长出"深度

每个图像像素 (u, v) 有一个 2D 特征向量 f（比如 256 维，来自 Backbone）。同时网络用一个小分支（DepthNet）预测这个像素的**深度分布**：把 [0, 51.2m] 分成 D=128 个区间，输出 128 个概率（这个像素"大概在哪个距离"）。

然后做**外积**：

```
f_c(u,v) × d(u,v) → 形状 [C, D]
```

> [!note] 直觉：一个像素变成"一串像素"
> 一个 2D 像素在 3D 世界里其实对应**一条射线上的无数个点**（[[3D视觉与投影几何]] 讲过）。Lift 这一步就是：把每个像素的特征，沿这条射线"拉伸"成 D 份，每份是"如果这个物体在这个深度，它应该长这样"。深度分布 d 决定每份的可信度。
> **关键参数**：D=128（nuScenes），范围 [0, 51.2m]，分辨率 0.4m。

### Step 2: Splat（拍平）——把 3D 特征"倒"进 BEV 网格

每个像素 × 每个深度 bin，就有一个 3D 空间点。用相机内外参把这些 3D 点变换到自车坐标系（ego），再找到它落在哪个 BEV 网格里，把特征**累加**进去：

```
for 每个像素 (u,v) × 每个深度 bin d:
    3D 点 = unproject(u, v, depth[d])   # 相机 → 世界坐标
    BEV 格 = world_to_bev(3D 点)         # 世界 → BEV 索引
    BEV[格] += 锥体特征[u,v,d]            # 累加（Voxel Pooling）
```

> [!note] 直觉：往"地面地图"上倒特征
> 想象把 3D 空间当成一个"杯子堆"，BEV 网格是地面的"格盘"。每个 3D 特征点落到格盘上的对应格子，累加起来。**同一个 BEV 格子可能收到很多像素的特征（累加），也可能一个都收不到（远处、稀疏区）**——这就是前面 [[BEV感知全景]] 提到过的"前向投影特征分布不均"问题。

### Step 3: Shoot（发射）——在 BEV 上做检测

BEV 特征出来后，后面就是标准的检测流程：BEV Backbone（ResNet 风格）再提特征，CenterPoint 检测头（热力图 + 回归）输出 3D 框。

> [!note] 名字彩蛋
> "Shoot"原意是"发射"，论文里指的是在 BEV 图上做规划（把路径射出去）。BEVDet 作为检测框架把最后一步换成了检测头——**LSS 的"通用 BEV 特征"可以接任何下游任务**，这是它生命力的来源。

---

## 二、BEVDet——工业级 LSS 基线

### 2.1 架构全景

```
Image Encoder:    ResNet-50 + FPN (多尺度特征)
                  输出: 各相机 [C=256, H/8, W/8]

View Transform:   LSS (DepthNet → 锥体 → Voxel Pooling)
                  输出: BEV [C, 128, 128]

BEV Encoder:      ResNet-style BEV Backbone
                  输出: BEV [C, 128, 128]

Detection Head:   CenterPoint (heatmap + regression)
```

**四个模块的分工**：
1. **Image Encoder**（图像编码器）：每个相机独立过 ResNet-50 + FPN，提取多尺度 2D 特征。
2. **View Transform**（视图变换）：LSS 核心，把 2D 特征搬成 BEV 特征（本篇第一节的三步）。
3. **BEV Encoder**（BEV 编码器）：在 BEV 特征上再做几层卷积，让特征在"地图空间"里交互（邻居格子互相"通气"）。
4. **Detection Head**（检测头）：CenterPoint 风格——先在 BEV 上预测目标中心的热力图（有没有东西、在哪），再回归 3D 框属性（尺寸、朝向、速度）。

> [!note] CenterPoint 为什么是标配检测头？
> BEV 上物体通常不重叠（俯视视角），"中心点检测"（CenterPoint）天然合适：预测每个格子的"是目标中心的概率"热力图，峰值即目标中心。它去掉了 anchor 设计（[[计算机视觉基础]] 里讲过的 anchor-free 思路），简单又准。

### 2.2 为什么 BEVDet 比 BEVFormer 快这么多？（工程视角，重要！）

| 算子 | BEVDet | BEVFormer |
|------|--------|-----------|
| 主要计算 | Conv 2D | Deformable Attention |
| CUDA 优化 | cuDNN 高度优化 | 自定义 CUDA kernel |
| TensorRT 支持 | 完美 (标准 ONNX ops) | 需要 custom plugin |
| INT8 量化 | 精度损失 <0.5% | 精度损失 2-3% |
| 推理 FPS (A100) | 15+ | 2.5 |

**核心原因**：Conv2D 是 GPU 上优化了 30+ 年的算子，cuDNN 把它榨到了接近硬件极限；而 Deformable Attention 的 **bilinear sampling 是随机访存模式**——每个采样点都要去特征图上的随机位置取数，GPU cache 命中率极低，访存成为瓶颈。

> [!warning] 这个对比的深层启示
> **模型的"精度优势"和"工程优势"经常是分离的。** BEVFormer 精度略高但部署困难（自定义 kernel、TensorRT 要写 plugin、INT8 量化掉点多）；BEVDet 精度略低但纯卷积、部署一条龙。**量产选型时，这个 trade-off 往往比精度差 5 个点更重要**——这也是为什么 BEVDet 被称为"工业级基线"。

---

## 三、BEVDepth——解决 LSS 的核心弱点

### 3.1 LSS 的深度估计到底差在哪？

LSS 的深度分布是**隐式学习**的——网络只靠检测 loss 的梯度"顺带"学深度，没有任何显式监督。这导致三个问题：

1. **远处深度模糊**：检测 loss 对远处物体的梯度弱 → 深度分布"摊得很平"（模型不敢确定多远）。
2. **遮挡区域偏近**：被遮挡像素的深度分布偏向近处——因为近处特征梯度更大，模型"偷懒"把概率压在近处。
3. **缺乏显式几何约束**：没有任何 3D 真值告诉网络"你猜的深度对不对"。

> [!note] 直觉：让一个没学过"距离感"的学生去估距离
> 只告诉他"检测错了"（检测 loss），他很难知道"错在深度"。**这就是"深度估计不准 → BEV 全歪"级联误差的根源**（[[3D视觉与投影几何]] 里讲过）。

**BEVDepth 的核心贡献**：用 LiDAR 点云提供**稀疏但准确的深度监督**——把"距离感"直接教给网络。

### 3.2 深度监督机制——"LiDAR 当老师"

```python
# 1. 将 LiDAR 点云投影到每个相机的图像平面
lidar_pts_3d = lidar_points  # [N_pts, 3] in ego frame
pts_2d = project_to_camera(lidar_pts_3d, K, RT)
# [N_pts, 2] (u, v) — 落在图像内的点

# 2. 对每个像素 (u, v) 如果有 LiDAR 投影点 → 有深度真值
depth_gt = compute_depth(lidar_pts_3d)  # [H, W] — 稀疏深度图

# 3. 深度监督 Loss
# 将深度真值转换为 one-hot distribution over D=128 bins
depth_gt_onehot = to_onehot(depth_gt, bins=128, range=[0, 51.2])
# [H, W, 128]

# Binary Cross-Entropy between predicted depth dist and GT one-hot
L_depth = BCE(depth_pred, depth_gt_onehot) * valid_mask
# valid_mask = 1 对于有 LiDAR 投影的像素

# 4. 总 Loss
L_total = L_det + λ_depth * L_depth  # λ_depth = 1.0 (from paper)
```

**三步走明白**：
1. **投影**：LiDAR 点云（知道每个点的精确 3D 坐标）用相机内外参投影到图像上，落在图像内的点，其深度就是"这个像素的真实距离"。
2. **构造真值**：把真实深度转成 one-hot（在 128 个 bin 里，真实深度所在的 bin 是 1，其他是 0），得到稀疏深度真值图。
3. **监督**：对预测的深度分布和真值算 BCE 损失，只算"有 LiDAR 命中的像素"（valid_mask）。

> [!note] 为什么用"one-hot 分类"而不是直接回归深度值？
> 因为 LSS 的深度本来就是"分布"（128 个 bin 的概率），用分类（BCE）监督分布、用回归监督数值都不对味。**监督"分布的形状"（峰值应在真实深度处）比监督"数值"更平滑、更好训**——和 [[3D视觉与投影几何]] 里"深度分布 vs 深度回归"的道理一致。

### 3.3 消融——深度监督的效果到底有多大？

| 配置 | NDS | mAP | 远处 AP |
|------|:---:|:---:|:-----:|
| BEVDet (无深度监督) | 48.8 | 42.4 | 基准 |
| + 深度监督 (BEVDepth) | **56.5** | **49.8** | +8.3% |
| + 深度修正网络 | 57.1 | 50.3 | +9.5% |

**NDS 从 48.8 → 56.5，+7.7 个点**——纯靠"给深度加监督"，不改变检测架构，就有如此巨大的提升。而且提升最大的是**远处物体**（+8.3%）——因为 LSS 在远距离深度估计最差，恰好是监督收益最大的地方。

> [!warning] 这个消融的启示
> **"瓶颈在哪，改进就在哪"**——LSS 的瓶颈是深度，所以深度监督收益巨大。读论文时养成习惯：先找它的"消融表里哪一项提升最大"，那一项往往就是该方法的真正贡献点。

### 3.4 追问：LiDAR 点云是稀疏的（~10% 像素有深度真值），够用吗？

够用。原因：
1. LiDAR 点覆盖了大部分重要区域（路面、障碍物表面）。
2. 无 LiDAR 投影的像素不产生深度 loss（`valid_mask=0`）→ 这些像素的深度**仍然从 detection loss 中隐式学习**（两种监督互补，不是二选一）。
3. 消融证据：只用 **10% 的随机深度真值** → NDS 仍提升 2.5 点 → **即使稀疏监督也大幅优于纯隐式学习**。

> [!note] 深层原因：监督的"锚点效应"
> 不需要所有像素都有真值——**只要一部分像素的深度是准的，网络就能学到"深度长什么样"（分布的形状、远近的规律），然后泛化到没真值的像素**。稀疏真值是"锚"，不是"全部答案"。

---

## 四、BEVPoolv2——视锥体投影的工程优化

### 4.1 标准 Voxel Pooling 的瓶颈

LSS 的 Splat 步骤里有个性能杀手：

```python
# 标准实现 (BEVDet):
for each pixel (u,v):
    for each depth bin d:
        3D point = unproject(u, v, depth[d])  # 相机 → 世界坐标
        bev_index = world_to_bev(3D_point)      # 世界 → BEV 索引
        bev[bev_index] += frustum_feat[u,v,d]   # scatter add → 慢!
```

问题：每个像素 × D=128 个深度 bin = **几百万次 scatter add**，而且是 **random memory access**（相邻像素可能落到相距很远的 BEV 格子）→ GPU cache 命中率极低，慢到成为整个框架的瓶颈。

### 4.2 BEVPoolv2 的优化思路

```python
# 加速策略 (CUDA kernel):
# 1. 预计算: 提前计算所有像素×深度 bin 对应的 BEV 索引 (grid lookup table)
#    这部分是确定的 (只依赖相机内外参，不依赖特征值) → 只算一次
precomputed_bev_indices = compute_bev_indices(cam_params, D=128, H, W)
# [6 cameras, D, H, W, 2] — (bev_x, bev_y)

# 2. 排序: 将像素按 BEV 索引排序 → 使同一个 BEV cell 的像素聚集
sorted_indices = sort_by_bev_index(precomputed_bev_indices)

# 3. 前缀和聚合: 相邻的同一 BEV cell 用 atomicAdd → 减少 scatter 开销
bev = prefix_sum_aggregate(frustum_feat, sorted_indices)

# 实测加速: 2-3× faster than naive voxel pooling
```

> [!note] 优化的本质：把"随机访存"变成"顺序访存"
> scatter add 慢在随机访问。BEVPoolv2 的思路是：**先按 BEV 索引排序，让"去同一个格子的像素"在内存里挨在一起**，再顺序累加——访存从"随机跳"变成"顺序走"，cache 友好，快 2-3 倍。
> **预计算**是另一个关键：像素→BEV 索引只依赖相机内外参（不动），训练/推理时不变，所以**只算一次存起来**，不用每帧重算。
>
> 这告诉我们：**工程优化往往不是"发明新算法"，而是"让访存模式匹配硬件"**。类似的思路贯穿整个 [[模型部署与延迟优化]]。

---

## 五、BEVDet4D——在不改框架的前提下加时序

```
当前帧 (t) BEV: B_t [128, 128, C]
上一帧 (t-1) BEV: B_{t-1} → align with ego-motion → warped B_{t-1}

融合: B_fused = Conv3D(concat(B_t, warped_B_{t-1}))
                           [128, 128, 2C] → [128, 128, C]
```

**关键设计**：
1. **Ego-motion 对齐**：上一帧的 BEV 特征按自车运动"搬"到当前坐标系（车动了，世界没动——[[BEV感知全景]] 里讲的时序优势）。
2. **用 3D Conv 融合**：把当前帧和上一帧拼接（通道翻倍），用 3D 卷积融合——**注意，这里选的是卷积而不是 Attention**！
3. **效果**：NDS +2.5（对比单帧 BEVDet），计算轻量、TensorRT 友好。

> [!note] BEVDet4D vs BEVFormer 的时序路线之争
> - BEVFormer：用 **Temporal Self-Attention** 做时序（query 和历史 BEV 特征做注意力）——灵活但贵。
> - BEVDet4D：用 **Conv3D** 做时序（两帧拼接后卷积）——简单便宜，部署友好。
> **同一个目标（利用时序），两种工程哲学**：BEVFormer 追求表达力，BEVDet4D 追求可部署性。这个对比值得反复品味——它是"论文派 vs 工程派"思维差异的缩影。

---

## 六、面试官追问——必背的三个问题

### Q1: BEVDet 和 BEVFormer 选哪个部署到 Orin？为什么？

选 BEVDet。理由：
1. Orin（30W TDP）算力约 A100 的 1/15 → BEVFormer 2.5 FPS on A100 = 0.17 FPS on Orin（不可用）。
2. BEVDet 纯 Conv → TensorRT INT8 优化后可达 10-15 FPS on Orin。
3. 但 BEVDet 精度低（NDS 48.8 vs 56.9）→ 可以用 **BEVDepth 补齐深度精度**（NDS 56.5）。

> [!note] 回答框架（面试可用）
> **先算账（算力预算），再谈精度（差距多大），最后给方案（BEVDet+深度监督）**——把"部署约束"放在"精度追求"前面，这是自动驾驶量产面试想看到的思维方式。

### Q2: BEVDepth 的深度监督要求 LiDAR，纯视觉方案怎么办？

纯视觉方案有两个替代路径：
1. **自监督深度**：用 Structure from Motion (SfM) 或 photometric consistency（光度一致性）生成稀疏深度伪标签。
2. **时序多帧约束**：多帧之间的一致性约束可以隐式监督深度（不需要显式深度真值）。

但当前（2024）纯视觉的深度估计在远距离（50m+）仍不如 LiDAR 监督 → **这也是为什么纯视觉 BEV 方案的远距离检测是主要难点**（也是特斯拉坚持"纯视觉 + 海量数据"赌注的争议点）。

> [!warning] 纯视觉的深层问题
> 深度监督的"老师"没了，纯视觉只能靠"猜 + 一致性约束"。数据再多，"猜"的上限也受物理信息缺失限制（[[3D视觉与投影几何]] 的"单目无解"问题）——**这就是纯视觉方案和"相机+LiDAR"方案的根本差距来源**。

---

## 七、常见误区速查

> [!warning] 新手最常踩的坑
> 1. **以为 LSS 是"网络直接输出 BEV"**——不是！它是"像素抬升 → 几何投影 → 累加"的**显式几何过程**，中间步骤（深度分布、视锥）都是可解释的。
> 2. **以为深度分布是"回归一个数"**——是 128 个 bin 的**概率分布**（分类问题），监督也要用分类损失（BCE）。
> 3. **混淆"深度监督"和"深度真值"**——BEVDepth 的深度真值来自 LiDAR 投影（稀疏），不是稠密深度图。
> 4. **以为 BEVDepth 只提升了远处**——远处提升最大（+8.3%），但近处整体 NDS 也涨了（48.8→56.5），是全局性的。
> 5. **忽略工程优化**——BEVPoolv2 这类优化才是 LSS 能"工业级"的原因；论文里看似"技术细节"的工程模块，往往是量产可行性的关键。

---

## ✅ 检验自己（自测题）

> [!question] Q1：用你自己的话讲一遍 LSS 的 Lift / Splat / Shoot 三步。
> 提示：分别对应"给像素长深度"、"倒进 BEV 网格"、"在 BEV 上做任务"。

> [!success]- 参考答案
> Lift：每个像素的特征和它预测的深度分布做外积，把 2D 特征沿射线"拉伸"成 [C, D] 的视锥特征（一个像素变成一串深度候选）。Splat：用相机内外参把每个"像素×深度"对应的 3D 点变换到自车坐标，落进 BEV 网格并累加（Voxel Pooling），得到 BEV 特征。Shoot：在 BEV 特征上做下游任务（BEVDet 里是 CenterPoint 检测）。核心是"显式深度 + 几何投影"，全程可解释。

> [!question] Q2：BEVDet 为什么在部署上比 BEVFormer 有巨大优势？具体列 3 点。
> 提示：从算子、工具链、量化三个角度。

> [!success]- 参考答案
> ① 主要算子是 Conv2D，cuDNN 优化成熟（30+ 年），Deformable Attention 是随机访存、cache 不友好且需要自定义 CUDA kernel；② TensorRT 对标准 ONNX 算子完美支持，BEVFormer 需要 custom plugin；③ INT8 量化 BEVDet 精度损失 <0.5%，BEVFormer 损失 2-3%。综合下来 A100 上 BEVDet 15+ FPS vs BEVFormer 2.5 FPS。

> [!question] Q3：LSS 的深度估计有三个问题，分别是什么？BEVDepth 怎么解决？
> 提示：远处模糊 / 遮挡偏近 / 无几何约束。

> [!success]- 参考答案
> ① 远处深度模糊：检测 loss 对远处梯度弱，深度分布摊平 → BEVDepth 用 LiDAR 真值监督让远处深度分布集中；② 遮挡区域偏近：近处梯度大，模型偷懒押近处 → 显式深度真值纠正；③ 无几何约束：没有任何 3D 监督 → 加深度监督 loss（L_total = L_det + λ·L_depth）。本质：把"隐式学深度"变成"显式监督深度"。

> [!question] Q4：LiDAR 深度真值只有约 10% 像素有值，为什么"够用"？
> 提示：锚点效应 + 双监督互补。

> [!success]- 参考答案
> ① LiDAR 点覆盖了路面和障碍物等关键区域；② 无真值的像素仍从检测 loss 隐式学习，两种监督互补；③ 稀疏真值有"锚点效应"——网络从部分准确深度学到深度分布的形状和规律，再泛化到无真值像素。消融证明：只用 10% 随机真值，NDS 仍 +2.5 点。

> [!question] Q5：BEVDet4D 和 BEVFormer 都做时序，选型差异的根源是什么？
> 提示：Conv3D vs Attention，表达力 vs 可部署性。

> [!success]- 参考答案
> BEVDet4D 用 Conv3D（拼接两帧 BEV 后卷积融合），BEVFormer 用 Temporal Self-Attention（query 和历史特征做注意力）。前者计算轻、TensorRT 友好，后者表达力强（能学任意跨帧关联）但贵。根源是工程哲学差异：BEVDet4D 追求"在 LSS 框架内加时序还不影响部署"，BEVFormer 追求"时序融合的表达上限"。量产选 BEVDet4D 路线，论文刷榜选 BEVFormer 路线。

---

## 🛠 动手练习

### 练习 1：手算一次 Splat（20 分钟）

给定 2 个像素，深度 bin 数 D=4（深度 0.5m / 1m / 1.5m / 2m），相机为简化正交投影（3D 点 = (u×depth, v×depth)）：
- 像素 A：特征 f_A=[1, 2]，深度分布 [0.1, 0.7, 0.2, 0.0]（最可能 1m）
- 像素 B：特征 f_B=[3, 4]，深度分布 [0.0, 0.2, 0.8, 0.0]（最可能 1.5m）

BEV 网格分辨率 1m，范围 [0,2]×[0,2]。手算：
1. 每个"像素×深度"组合的 3D 坐标和 BEV 格子。
2. 每个格子累加的最终特征值。

> [!tip] 做完后自问
> ① 哪个像素对哪个格子贡献最大？② 如果像素 B 的深度分布改成 [0.25,0.25,0.25,0.25]（完全不确定），BEV 特征会怎样？（摊平 → 每个格子都有一点但都不强——这就是"深度模糊"的直观后果。）

### 练习 2：用 numpy 实现简化版 Splat（进阶，30-60 分钟）

```python
import numpy as np

def simple_splat(features, depth_probs, cam_params, bev_size=128, bev_range=51.2):
    """简化版 Splat：把 [H, W, C] 特征 + [H, W, D] 深度分布投影到 BEV。
    这里用最简假设：像素 (u,v) 深度 d 的 3D 点是 (u·d, v·d)，忽略真实内外参。"""
    H, W, C = features.shape
    D = depth_probs.shape[-1]
    depths = np.linspace(0.1, bev_range, D)          # D 个深度候选值
    bev = np.zeros((bev_size, bev_size, C))

    for u in range(H):
        for v in range(W):
            for d_idx in range(D):
                z = depths[d_idx]
                x = u * z / bev_range * bev_size       # 简化投影
                y = v * z / bev_range * bev_size
                bx, by = int(x), int(y)
                if 0 <= bx < bev_size and 0 <= by < bev_size:
                    # 特征 × 深度概率，累加进 BEV 格子
                    bev[bx, by] += features[u, v] * depth_probs[u, v, d_idx]
    return bev

# 造一个假输入跑一下，观察 BEV 的稠密/稀疏分布
```

> [!tip] 做完后自问
> ① 哪里的 BEV 格子最"挤"（收到很多特征）？哪里最空？② 这和真实 LSS 的"近处密、远处疏"一致吗？③ 想想真实实现为什么要用 BEVPoolv2 优化——这个三重循环跑起来有多慢？

### 练习 3：跑通 BEVDet 推理（可选，1-2 天）

按 BEVDet 官方仓库（GitHub: HuangJunJie2017/BEVDet）在 nuScenes mini 上跑通推理：
1. 下载 nuScenes mini 数据，配置环境。
2. 用官方权重跑一次推理，可视化 BEV 检测结果。
3. 修改 `bev_size` 从 128 → 256，对比精度（NDS/mAP）变化。

> [!tip] 这是新手路径 Phase 6 的正式任务，[[学习计划]] 里有对应勾选项。跑不通别灰心——环境问题是每个 BEV 新手的第一道坎，把报错记录下来（[[训练排错实战手册]] 有排查思路）。

---

## ➡️ 下一步学什么

按知识库学习路径，读完本篇你应该接着：

1. **[[BEVFormer详解]]** —— 对照学习：同样的 BEV 目标，Transformer 派怎么做（新手路径第 3 站）。
2. **[[占据网络与GOD]]** —— BEV 检测之后：从"地面地图"升级到"3D 空间"（新手路径第 4 站）。
3. **[[PETR系列]]** —— 进阶：第三条路线（3D 位置编码）怎么绕过显式投影。
4. **[[模型部署与延迟优化]]** —— 想深入"BEVDet 为什么部署友好"时看。

> 💡 建议带着这张对比表去读 [[BEVFormer详解]]：LSS（显式深度+拍平）vs BEVFormer（隐式注意力+采样）——两条路线的差异会在下一篇全部浮现。

---

## 相关笔记

- [[BEV感知全景]] — BEV 整体认知与四条路线
- [[BEVFormer详解]] — Transformer 派 BEV 感知
- [[3D视觉与投影几何]] — LSS 的数学基础（投影、坐标系）
- [[PETR系列]] — 3D 位置编码路线
- [[占据网络与GOD]] — BEV → 3D 占据升级
- [[模型部署与延迟优化]] — 部署视角的算子分析
- [[AI-Infra详解]] — 训练与推理优化
