---
tags: [BEV, detection, LSS]
created: "2026-07-21"
updated: "2026-07-28"
---

# BEVDet 与 BEVDepth

> **核心范式**: Lift-Splat-Shoot (LSS) — 显式估计深度分布，将 2D 特征提升到 3D 并拍平到 BEV。
> BEVDet 建立了 LSS 范式的工业级 Baseline，BEVDepth 通过深度监督解决了 LSS 的核心弱点。

---

## 一、LSS 范式回顾

```
Lift:   像素特征 × 深度分布 (D bins) → 3D 视锥体特征
        每个像素 (u,v): f_c(u,v) × d(u,v) → [C, D] 的锥体特征
Splat:  锥体特征 → 利用相机内外参变换到 ego 坐标系 → Sum/Voxel Pooling → BEV
Shoot:  BEV 特征 → BEV Backbone (ResNet) → Detection Head (CenterPoint)
```

**关键参数**:
- 深度分布 bins: $D=128$ (nuScenes), 范围 $[0, 51.2m]$, 分辨率 $0.4m$
- 锥体特征: $C \times D \times H \times W$ — 对 6 相机 × 640×960 下是巨大的
- BEV 网格: $128 \times 128$ (BEVDet), $256 \times 256$ (高分辨率变体)

---

## 二、BEVDet — 工业级 LSS 基线

### 2.1 架构

```
Image Encoder:    ResNet-50 + FPN (多尺度特征)
                  输出: 各相机 [C=256, H/8, W/8]

View Transform:   LSS (DepthNet → 锥体 → Voxel Pooling)
                  输出: BEV [C, 128, 128]

BEV Encoder:      ResNet-style BEV Backbone
                  输出: BEV [C, 128, 128]

Detection Head:   CenterPoint (heatmap + regression)
```

### 2.2 为什么 BEVDet 比 BEVFormer 快？

| 算子 | BEVDet | BEVFormer |
|------|--------|-----------|
| 主要计算 | Conv 2D | Deformable Attention |
| CUDA 优化 | cuDNN 高度优化 | 自定义 CUDA kernel |
| TensorRT 支持 | 完美 (标准 ONNX ops) | 需要 custom plugin |
| INT8 量化 | 精度损失 <0.5% | 精度损失 2-3% |
| 推理 FPS (A100) | 15+ | 2.5 |

**核心原因**: Conv2D 是 GPU 上最优化过的算子，30+ 年的优化历史。Deformable Attention 的 bilinear sampling 是随机访存模式，cache 命中率低。

---

## 三、BEVDepth — 解决 LSS 的核心弱点

### 3.1 LSS 的深度估计问题

LSS 的深度分布是**隐式学习**的 — 只有检测 loss 反向传播。问题：

1. **远处深度模糊**: 检测 loss 对远处物体的梯度弱 → 深度分布分散
2. **遮挡区域偏近**: 被挡像素的深度分布会偏向近处（近处特征梯度和更大）
3. **缺乏显式几何约束**: 没有任何 3D 几何监督

**BEVDepth 的核心贡献**: 用 LiDAR 点云提供**稀疏但准确的深度监督**。

### 3.2 深度监督机制

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

### 3.3 消融 — 深度监督的效果

| 配置 | NDS | mAP | 远处 AP |
|------|:---:|:---:|:-----:|
| BEVDet (无深度监督) | 48.8 | 42.4 | 基准 |
| + 深度监督 (BEVDepth) | **56.5** | **49.8** | +8.3% |
| + 深度修正网络 | 57.1 | 50.3 | +9.5% |

深度监督对**远处物体**的提升最大（因为 LSS 在远距离深度估计最差）。

### 3.4 追问: LiDAR 点云是稀疏的（~10% 像素有深度真值），够用吗？

够用。原因：
1. LiDAR 点覆盖了大部分重要区域（路面、障碍物）
2. 无 LiDAR 投影的像素不产生深度 loss（`valid_mask=0`）→ 这些像素的深度仍然从 detection loss 中隐式学习
3. 消融: 只用 10% 的随机深度真值 → NDS 仍提升 2.5 点 → 证明即使是稀疏深度监督也大幅优于纯隐式学习

---

## 四、BEVPoolv2 — 视锥体投影优化

### 4.1 标准 Voxel Pooling 的瓶颈

```python
# 标准实现 (BEVDet):
for each pixel (u,v):
    for each depth bin d:
        3D point = unproject(u, v, depth[d])  # 相机 → 世界坐标
        bev_index = world_to_bev(3D_point)      # 世界 → BEV 索引
        bev[bev_index] += frustum_feat[u,v,d]   # scatter add → 慢!
```

问题：每个像素 × D=128 个深度 bin → scatter add 是 random memory access → GPU cache 命中率极低。

### 4.2 BEVPoolv2 优化

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

---

## 五、BEVDet4D — 时序扩展

在不改变 LSS 框架的前提下加入时序：

```
当前帧 (t) BEV: B_t [128, 128, C]
上一帧 (t-1) BEV: B_{t-1} → align with ego-motion → warped B_{t-1}

融合: B_fused = Conv3D(concat(B_t, warped_B_{t-1}))
                           [128, 128, 2C] → [128, 128, C]
```

关键设计：用 3D Conv 而非 Attention 做融合 → 计算轻量，TensorRT 友好。

效果: NDS +2.5 (vs single-frame BEVDet)

---

## 六、面试官追问

### Q: BEVDet 和 BEVFormer 选哪个部署到 Orin？为什么？

选 BEVDet。理由：
1. Orin (30W TDP) 算力约 A100 的 1/15 → BEVFormer 2.5 FPS on A100 = 0.17 FPS on Orin (不可用)
2. BEVDet 纯 Conv → TensorRT INT8 优化后可达 10-15 FPS on Orin
3. 但 BEVDet 精度低 (NDS 48.8 vs 56.9) → 可以用 BEVDepth 补齐深度精度 (NDS 56.5)

### Q: BEVDepth 的深度监督要求 LiDAR，纯视觉方案怎么办？

纯视觉方案有两个替代路径：
1. **自监督深度**: 用 Structure from Motion (SfM) 或 photometric consistency 生成稀疏深度伪标签
2. **时序多帧约束**: 多帧之间的一致性约束可以隐式监督深度（不需要显式深度真值）

但当前 (2024) 纯视觉的深度估计在远距离 (50m+) 仍不如 LiDAR 监督 → 这也是为什么纯视觉 BEV 方案的远距离检测是主要难点。

---

> 📚 **相关**: [[BEVFormer详解]], [[BEV感知全景]], [[AI-Infra详解]]
