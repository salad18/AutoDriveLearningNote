---
title: "源码走读指南：BEVFormer × mmdet3d"
tags: [practice, mmdet3d, BEVFormer, source-reading]
created: "2026-11-04"
updated: "2026-11-04"
---

# 源码走读指南：BEVFormer × mmdet3d

> 一句话导读：面试官问"你会 mmdet3d 吗"，不是问"你跑过 demo"，而是问"**你读过训练/推理全链路吗**"——数据怎么进、backbone 怎么出、head 怎么算、loss 怎么配、结果怎么评估。本篇给你一条**可执行的走读路线**：14 天、7 个检查点，边读边答边记，读完你能画出完整数据流张量图。

> ⚡ **面试前只看速答版** → [[开源栈追问速答卡-mmdet3d]]（7 检查点 + 12 追问，30 秒答法）；本篇用于**真正走读**。

> [!warning] 诚实提示
> 本篇是**走读指南**（路线图 + 检查点 + 问题清单），不是替代动手的教程——真正的收获来自你在本地把仓库跑起来、把张量形状打出来。环境配置参考 [[本地训练方案]] / [[训练排错实战手册]]；代码细节参考 [[BEVFormer详解]] 第 6 节。

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | 为什么面试看重源码级理解、环境策略、mmdet3d 目录地图、BEVFormer 全链路 7 个检查点、面试问答、14 天时间表 |
| **需要的前置知识** | [[BEVFormer详解]]（架构+代码片段）、[[模型部署与延迟优化]]（部署视角）、[[本地训练方案]]（环境）|
| **学完之后你能** | ① 画出 BEVFormer 从图像到检测输出的完整数据流；② 说出每个模块的输入输出形状；③ 回答源码级追问（配置文件/Sampler/后处理/评估）|
| **预计阅读时间** | 20 分钟读本篇 + 14 天走读 |

> [!tip] 走读的姿势：**从形状出发，不从代码行出发**
> 别逐行读。先找每个中间张量的**形状**——形状就是模块的"身份证"。`[B, N_q, C]` 是什么？`[B, N_q, n_heads, n_levels, n_points, 2]` 里的 2 是什么？**形状看懂，逻辑自明**。每读一个模块，先问"输入形状、输出形状、这个变换在物理上干什么"。

---

## 一、为什么面试看重源码级理解

| 面试官的意图 | 源码理解能证明什么 |
|-------------|-------------------|
| "你真的用过 mmdet3d 吗" | 知道 config 和代码怎么对应，不是只会 pip install |
| "出问题你能自己修吗" | 能定位到具体模块（loss NaN 在哪层、OOM 在哪步） |
| "你理解设计取舍吗" | 知道为什么这个模块写成这样（如 Sampler 为什么放哪层） |
| "你能改模型吗" | 能加 head / 换 backbone / 改 loss 而不看文档 |

> JD 原文（2025-2026 常见表述）："熟悉 mmdetection3d / OpenPCDet 源码"——**这就是为什么这篇是 🔥 高优先**（见 [[查漏补缺与求职行动清单]]）。

---

## 二、环境策略（别在环境上耗 3 天）

```
推荐优先级:
① 官方 Docker（如果有）—— 最省心
② 官方 requirements + conda 锁版本 —— 按 README 的版本装，别追新
③ 最小推理环境（只跑推理不训）—— 先跑通再说
④ 纯读代码（无环境）—— 用 GitHub 网页版读，也能走完 7 个检查点
```

> [!warning] 常见坑
> BEVFormer 老代码 + 新 torch 是"地狱级配置"：CUDA 版本、mmcv-full 编译、自定义 CUDA op 编译。**先接受"环境是坑"，把时间花在读懂上**——读代码不需要 GPU，4 个检查点里 3 个纯读就能完成。

---

## 三、mmdet3d 目录地图（先认路）

```
mmdet3d/
├── configs/                    # 所有模型的配置文件（模型定义 = config！）
│   └── bevformer/              # BEVFormer 的 config
├── mmdet3d/
│   ├── models/
│   │   ├── backbones/          # ResNet / VovNet / Swin
│   │   ├── necks/              # FPN
│   │   ├── dense_heads/        # 检测头（BEVFormer Head 相关）
│   │   └── detectors/          # 单阶段/两阶段检测器组装
│   ├── datasets/               # 数据加载 + pipeline（阅读重点！）
│   ├── core/                   # hook / bbox assigner / sampler
│   └── apis/                   # train / inference 入口
└── projects/BEVFormer/         # BEVFormer 官方实现（插件式）
    └── projects/mmdet3d_plugin/bevformer/
        ├── bevformer_head.py           # Head
        ├── spatial_cross_attention.py  # 空间交叉注意力
        ├── temporal_self_attention.py  # 时序自注意力
        └── transformer.py              # Encoder 组装
```

> [!note] 关键认知：mmdet3d 里"模型 = config"
> 你看 `configs/xxx.py` 的 `model = dict(type='BEVFormer', ...)`——**config 就是模型的"装配图"**。读源码前先读 config：数据 pipeline、backbone、neck、head、loss 都在 config 里声明。**面试问"怎么改模型"，标准答案是"改 config + 对应代码"**。

---

## 四、7 个检查点（走读路线，按这个顺序）

### 检查点 0：数据 pipeline（先读数据）

- **文件**：`configs/` 里的 `data` 段 + `mmdet3d/datasets/pipelines/`（`loading.py`、`transforms_3d.py`、`compose.py`）
- **要答的问题**：
  1. 一张图从磁盘到模型输入，经过哪些变换？（加载 → resize → 归一化 → 多相机打包 → 时序采样）
  2. `LoadMultiViewImageFromFiles` 干了什么？6 个相机怎么拼成 batch 维？
  3. **时序帧怎么采**（`LoadMultiViewImageFromFiles` + `prev/next` 帧）？为什么 BEVFormer 需要 3 帧历史？
- **产出**：画一张"单样本数据流"图（shape 标注）。

### 检查点 1：Backbone + Neck（图像特征）

- **文件**：`mmdet3d/models/backbones/`（ResNet）+ `necks/fpn.py`
- **要答的问题**：6 个相机 × T 帧的输入形状 → backbone 输出什么形状？FPN 输出几层（多尺度）？
- **产出**：张量形状链（如 `[B*6*T, 3, H, W]` → `[B*6*T, C, H/8, W/8]` × 4 层）。

### 检查点 2：BEV Queries 初始化（Head 入口）

- **文件**：`bevformer_head.py` 的 `__init__` / `get_bev_queries`
- **要答的问题**：BEV 网格 200×200 怎么变成 40000 个 query？`bev_positions`（坐标）和 `bev_embedding`（可学习）分别是什么？（[[BEVFormer详解]] 6.2 有完整代码）
- **产出**：写出 query 的形状 `[B, 40000, 256]` 和来源。

### 检查点 3：Spatial Cross-Attention（核心！）

- **文件**：`spatial_cross_attention.py` 的 `forward` + `point_sampling`
- **要答的问题**：
  1. 每个 BEV query 的 3D 参考点怎么算（4 个高度 × 4 采样点）？
  2. `reference_points_cam` 怎么投影的（内参/外参矩阵）？
  3. FOV mask 怎么过滤"落在图外"的点？
  4. `sampling_offsets` 形状 `[B, N_q, n_heads, n_levels, n_points, 2]` 里最后的 2 是什么？（偏移量 Δx, Δy）
- **产出**：把 [[BEVFormer详解]] 6.3 的每一步在代码里找到对应函数，标行号。

### 检查点 4：Temporal Self-Attention（时序融合）

- **文件**：`temporal_self_attention.py`
- **要答的问题**：
  1. 上一帧 BEV 特征怎么 warp 到当前帧（ego-motion）？
  2. 时序 mask 为什么需要（新出现的区域 vs 历史）
  3. 时序融合的输入形状（`[B, 40000, 256]` × 2 帧）
- **产出**：画出"上一帧 BEV → warp → 与当前帧拼接"的数据流。

### 检查点 5：Head 输出 + Loss（怎么算分）

- **文件**：`bevformer_head.py` 的 `forward` + `loss` 函数 + config 的 `loss` 段
- **要答的问题**：
  1. 40000 个 query 怎么变成检测结果（分类分数 + 回归框 + 速度）？
  2. 分类用 Focal Loss、回归用 L1？权重多少？（`λ=2.0`）
  3. 正负样本怎么配（assigner：哪些 query 算正样本）？
- **产出**：写出 loss 公式 + config 里的权重。

### 检查点 6：后处理 + 评估（输出到分数）

- **文件**：`apis/`、`mmdet3d/core/`、`mmdet3d/datasets/`（评估器）
- **要答的问题**：
  1. 网络输出 → NMS → 检测框，中间做了什么（top-k、阈值、NMS）？
  2. nuScenes 评估怎么算 NDS/mAP（[[常见面试题-感知算法]] Q9）？
  3. 推理时 batch=1 的流程和训练有什么不同（BN/Dropout 行为）？

---

## 五、产出：完整数据流图（走完 7 个检查点后画这个）

```
输入: 6 相机 × 3 帧图像 + 相机参数 + ego-motion
  │
  ▼
[Backbone+FPN]: 每相机每帧 [3, H, W] → 4 层多尺度特征 [C, H/8, W/8] ×4
  │
  ▼
[BEV Queries]: [40000, 256]（坐标 + 可学习嵌入）
  │
  ▼
[Spatial Cross-Attention]: 每个 query 采样 16 个 3D 点 → 投影到 6 相机 → FOV 过滤 → 加权
  │
  ▼
[Temporal Self-Attention]: 与 warp 后的上一帧 BEV 融合
  │
  ▼
[Head]: 分类 + 回归（框/速度）
  │
  ▼
[后处理]: top-k → NMS → 检测输出
  │
  ▼
[评估]: NDS/mAP（nuScenes val）
```

> **能画出这张图 + 标出每步 shape，你的源码理解就过关了。** 面试时直接给面试官画。

---

## 六、面试问答（源码级追问）

> [!question] Q1：BEVFormer 的模型定义在哪里？怎么加一个新的检测头？

> [!success]- 参考答案
> 模型定义在 **config 文件**（`model = dict(type='BEVFormer', ...)`）——mmdet3d 里"模型 = config 装配图"。加新检测头：① 写新的 Head 类；② 在 config 里把 `bbox_head` 换成新类；③ 注册到 `__init__.py`。改模型的标准答案永远是"改 config + 写/改对应类"。

> [!question] Q2：正负样本是怎么分配的？

> [!success]- 参考答案
> 通过 `bbox_assigner`（如 `HungarianAssigner`/`MaxIoUAssigner`）——把 GT 框和预测 query 匹配：匈牙利匹配（最优二分匹配）或 IoU 阈值匹配。BEVFormer 用匈牙利匹配（类似 DETR），把 40000 个 query 和 GT 做匹配，匹配上的算正样本。config 里 `train_cfg` 的 `assigner` 就是它。

> [!question] Q3：为什么自定义 CUDA op（Deformable Attention）会让部署变难？

> [!success]- 参考答案
> `bev_pool`/`deformable_attn` 是自定义 CUDA 算子，ONNX/TensorRT 没有原生支持 → 需要写 plugin，而自定义 plugin 的优化程度远不如标准算子（[[常见面试题-感知算法]] Q5 提到实测：Orin 上量化后 3-5 FPS）。工程上通常：量化 → 算子融合 → 特殊算子单独实现。

> [!question] Q4：训练时 BN 和推理时有什么区别？为什么 BEV 模型常改 SyncBN？

> [!success]- 参考答案
> 训练用 batch 统计并更新 running mean/var，推理用 running 统计（固定）。BEV 模型输入大、显存紧（batch 4-16/GPU），BN 统计不稳定 → 用 SyncBN（跨卡同步统计）或 LN/GN。这也是"部署时 BN 可以融合进卷积"的由来（[[模型部署与延迟优化]]）。

---

## 七、14 天时间表

| 天 | 任务 | 检查点 |
|----|------|--------|
| 1-2 | 配环境（Docker/conda），跑通官方 demo 推理 | 0 |
| 3-4 | 读数据 pipeline + backbone/FPN | 0-1 |
| 5-6 | 读 BEV Queries + Spatial Cross-Attention | 2-3 |
| 7-8 | 读 Temporal Self-Attention + Head/Loss | 4-5 |
| 9 | 读后处理 + 评估 | 6 |
| 10 | 画完整数据流图（标 shape） | 全部 |
| 11-12 | 跑 nuScenes mini 推理，可视化 BEV 结果 | 全部 |
| 13-14 | 写走读笔记（7 个检查点答案）→ 进 [[面试复盘记录]] | 全部 |

> 💡 **有 GPU 版**（可选）：改 `bev_h/bev_w` 为 100×100 重新跑，观察 NDS 变化——用实验验证"分辨率 vs 精度/速度"的权衡（[[BEVFormer详解]] 练习 3）。

---

## ✅ 走读自测（全部能答才叫"读过"）

> [!question] 自测清单
> ① 一张图到模型输入经过哪些步骤？② 6 相机 × 3 帧的形状怎么变？③ 40000 个 BEV query 从哪来？④ 每个 query 采样几个 3D 点、怎么投影、怎么过滤？⑤ 上一帧怎么 warp 融合？⑥ loss 怎么配（公式+权重）？⑦ 输出到 NDS 经过什么？

> [!success]- 参考答案
> 全部能脱口而出 = 源码级理解。答不出其中任一 = 回到对应检查点重读。

---

## ➡️ 下一步学什么

1. **[[查漏补缺与求职行动清单]]** —— 🔥 第 2 项（源码走读）✅ 完成指南后，接着 🔥 第 4 项（TensorRT 部署实操）。
2. **[[BEVFormer详解]]** —— 第 6 节代码细节（本篇是路线图，它是内容）。
3. **[[训练排错实战手册]]** —— 环境/训练问题排查。
4. **[[模型部署与延迟优化]]** —— 源码理解后的部署视角。
5. **OpenPCDet 源码走读**（待建）—— 如果目标 LiDAR 感知岗，同理走 OpenPCDet（结构类似：数据/backbone/head/loss/评估）。

---

## 相关笔记

- [[BEVFormer详解]] — 架构与代码片段（本篇内容来源）
- [[查漏补缺与求职行动清单]] — 🔥 第 2 项（本篇是其落地指南）
- [[本地训练方案]] — 环境方案
- [[训练排错实战手册]] — 训练排查
- [[模型部署与延迟优化]] — 部署视角
- [[常见面试题-感知算法]] — Q5/Q6/Q9 源码相关面试题
- [[面试复盘记录]] — 走读笔记归档
