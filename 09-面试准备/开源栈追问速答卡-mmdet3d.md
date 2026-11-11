---
tags: [interview, 源码, mmdet3d, openpcdet, 速答]
created: "2026-11-11"
updated: "2026-11-11"
---

# 开源栈追问速答卡（mmdet3d / OpenPCDet）

> 面试官不会让你"讲一遍源码"，他会**挑一行问你是干嘛的**。这份卡片按"他会怎么问 → 30 秒怎么答 → 他再追什么"组织，**7 个检查点 + 12 个追问**，看完能接住"你读过源码吗"。
> 系统走读（14 天）→ [[源码走读指南-BEVFormer]]。

---

## 〇、先看穿：面试官只有四种问法

```
① "你读过 mmdet3d 源码吗？"          → 别答"读过"，答"我从数据流读了 7 个检查点"
② "BEVFormer 数据从哪进、到哪出？"    → 一张 shape 链
③ "这一行/这个类是干嘛的？"           → 定位到文件 + 一句话职责
④ "如果让你改 X，你动哪？"            → config + registry + 改哪个 forward
```

> [!tip] 现场看代码的 5 个动作（被共享屏幕时用）
> ① 先打开 **config**（`model = dict(type=...)` 就是装配图）；② `grep -rn "class XxxHead"` 找定义；③ 只追 **forward 的输入输出 shape**，不追实现细节；④ 看 **registry 注册**（它怎么被 config 找到）；⑤ 看 **loss / assigner / postprocess** 三段收尾。

---

## 一、检查点 0：数据 pipeline

**他会问**：一张图从磁盘到模型输入，经过哪些变换？

**30 秒答**：
`NuScenesDataset` 读 pkl → `Pipeline`（`LoadMultiViewImageFromFiles` 加载 6 相机 → resize/crop → `Normalize` → `Pad`/`DefaultFormatBundle` → `Collect3D` 打包）→ dataloader。时序帧由 config 的 `queue_length` 控制，`prev/next` 采样。

**追问**：
- 6 个相机怎么进 batch？→ 打包成 `[B, N_cam, 3, H, W]`，进 backbone 前摊平成 `[B*N_cam, 3, H, W]`。
- 时序怎么采？→ `queue_length`（BEVFormer 常用 3），取当前帧 + 历史帧的 BEV/图像。
- `Collect3D` 干嘛？→ 把 img/metas/label 组织成模型 forward 能吃的 dict。

**坑**：`LoadMultiViewImageFromFiles` **只负责读图和拼相机维**，不做几何变换——别把 resize 说成它干的。

---

## 二、检查点 1：Backbone + Neck

**他会问**：backbone 输出什么形状？

**30 秒答**：
`ResNet`（`mmdet3d/models/backbones/resnet.py`）接 `[B*N, 3, H, W]` → 4 个 stage，输出 `[B*N, C_i, H/2^{i+2}, W/...]`；`FPN`（`necks/fpn.py`）融合成多尺度 `[B*N, 256, H/8, W/8]` 等 4 层；再 reshape 回 `[B, N, 256, H/8, W/8]` 给 head。

**追问**：
- 为什么多尺度？→ 小目标靠高分辨率层。
- 6 相机共享 backbone 吗？→ 共享（同一套权重），所以能摊平进 batch。

**坑**：BEVFormer 的 backbone 输入是 **B×N 摊平**的，不是 6 个独立 batch——被问"显存怎么算"时这是关键。

---

## 三、检查点 2：BEV Queries 初始化

**他会问**：40000 个 query 怎么来的？

**30 秒答**：
BEV 网格 `200×200` → `200*200 = 40000` 个 query。`BEVFormerHead.get_bev_queries()` 返回两样：`bev_embedding`（可学习，`[B, 40000, 256]`）和 `bev_positions`（网格坐标 + 位置编码，告诉模型"这个 query 在世界系的哪个位置"）。

**追问**：
- 为什么要有 `bev_positions`？→ BEV query 本身没有空间含义，坐标编码给它"我在哪"。
- 分辨率能改吗？→ 能，改 config 的 `bev_h`/`bev_w`，但 query 数平方增长、注意力成本暴涨。

**坑**：`bev_embedding` 是**可学习参数**（所有样本共享），`bev_positions` 是**按网格算出来的**（每个样本可不同）——别混。

---

## 四、检查点 3：Spatial Cross-Attention（核心）

**他会问**：BEV query 怎么从 6 个相机"采"到特征？

**30 秒答**：
每个 BEV query 先在 **4 个高度**上取 3D 参考点 → 用内外参投影到 6 个相机 → 用 **FOV mask** 过滤落在图外的点 → 对命中的相机做 **deformable 采样**（每个参考点采 `n_points` 个偏移点）→ 加权求和回 BEV query。对应 `spatial_cross_attention.py` 的 `point_sampling` + `MSDeformableAttention3D`。

**追问**：
- `reference_points_cam` 形状？→ `[B, N_q, N_cam, 4, 2]`（4 个高度 × 2D 坐标）。
- FOV mask 干嘛？→ 标记"该 query 在该相机是否可见"，不可见的置零，避免采到图外。
- 为什么要 4 个高度？→ 单目深度不确定，多个高度假设覆盖不同距离/高度。

**坑**：投影用的是**每个相机的内外参**（`img_metas` 里的 `lidar2img`），不是一套全局矩阵——6 个相机各一套。

---

## 五、检查点 4：Temporal Self-Attention

**他会问**：上一帧 BEV 怎么对齐到当前帧？

**30 秒答**：
用自车位姿差（ego-motion）把上一帧 BEV **平移/旋转对齐**（代码里是 `shift` + grid_sample），再和当前帧 BEV 拼在一起做 deformable self-attention；新出现的区域用 **mask** 屏蔽（没有历史信息）。

**追问**：
- 不对齐会怎样？→ 时序特征错位，运动目标"重影"。
- 为什么用 self-attention 而不是直接 concat？→ 让每个 query 自己决定"看历史哪个位置"。

**坑**：对齐依赖**位姿精度**；标定/定位漂移会直接体现为时序融合变差。

---

## 六、检查点 5：Head 输出 + Loss

**他会问**：40000 个 query 怎么变成检测结果和 loss？

**30 秒答**：
`BEVFormerHead.forward` 用 `reg_branch`（回归中心/尺寸/朝向/速度）和 `cls_branch`（分类）两个 MLP 出预测；训练时用 **Hungarian assigner** 做一对一匹配（cost = 分类 + L1），分类用 **Focal Loss**（weight≈2.0），回归用 **L1**（weight≈0.25），具体以 config 为准。

**追问**：
- 为什么用匈牙利而不是 anchor？→ DETR 式集合预测，无需 NMS、无需预设 anchor。
- 正负样本怎么定？→ assigner 按最小代价一对一分配，其余为负。
- 速度怎么回归？→ 单独一个回归分支（nuScenes 需要 velocity）。

**坑**：Focal + L1 的权重是**调过的**，别自己改完说"论文这么写的"——以 config 为准。

---

## 七、检查点 6：后处理 + 评估

**他会问**：网络输出怎么变成 NDS？

**30 秒答**：
`get_bboxes`：按 `score_threshold` 过滤 → `max_num` 截断 →（可选）NMS → 输出框；`NuScenesDataset.evaluate` 调官方 devkit 算 mAP/NDS（mAP + 5 项误差）。

**追问**：
- 推理时和训练有什么不同？→ `model.eval()`（BN 用 running stats、关 dropout）、`with torch.no_grad()`。
- NDS 怎么算？→ `NDS = (5·mAP + mATE + mASE + mAOE + mAVE + mAAE)/10`。

**坑**：评估必须用**官方 devkit**，自己写的 mAP 和榜单对不上。

---

## 八、OpenPCDet 对照（面激光雷达公司时用）

| 环节 | mmdet3d | OpenPCDet |
|------|---------|-----------|
| 配置 | Python config | YAML config |
| 数据 | `NuScenesDataset` + pipeline | `DataProcessor` + `data_augmentor` |
| 体素化 | 少用 | `VoxelGenerator`（spconv） |
| Backbone | ResNet/FPN | `PointPillarScatter` / sparse conv + `BaseBEVBackbone` |
| Head | DETR 式 query | `AnchorHeadSingle` / `CenterHead` |
| 后处理 | top-k + NMS | NMS（`post_processing`） |

**一句话**：**mmdet3d 是"config 驱动 + 多模态"，OpenPCDet 是"YAML + 纯点云流水线"**；面试问"你用过哪个"，说清差异比背 API 有用。

---

## 九、"改模型"的标准答法（高频）

```
① 改 config：换 backbone / 改 loss 权重 / 改 bev 分辨率
② 注册新模块：继承 BaseModule + @MODELS.register_module()（老版本 @DETECTORS.register_module()）
③ 改 forward：加输出头 / 改采样点
④ 改评测：新任务要加 metric
⑤ 跑通：单卡小数据 → 多卡 → 全量
```

> [!warning] 别说"我直接改源码"
> 面试官想听的是"**config + 注册 + 最小改动**"，直接改源码 = 不可复现、不可维护。

---

## 十、12 个高频追问（一句话）

| # | 追问 | 一句话 |
|:--:|------|--------|
| 1 | config 怎么变成模型？ | `build_from_cfg` 按 `type` 查 registry 实例化 |
| 2 | 为什么用 config 不用硬编码？ | 实验管理/复现/批量扫参 |
| 3 | `register_module()` 作用？ | 把类注册进 registry，config 才能按名字找到 |
| 4 | pipeline 的 `Compose` 顺序？ | 加载→几何→归一化→打包，顺序错 shape 就错 |
| 5 | 多相机怎么打包？ | `[B, N, 3, H, W]`，backbone 前摊平 |
| 6 | 40000 query 怎么来？ | 200×200 BEV 网格 |
| 7 | FOV mask 作用？ | 过滤投影到图外的参考点 |
| 8 | prev_bev 怎么对齐？ | 位姿差 shift + grid_sample |
| 9 | assigner 怎么定正样本？ | 匈牙利一对一最小代价 |
| 10 | 训练/推理 BN 差异？ | eval 用 running stats |
| 11 | 改 num_classes 动哪？ | config + dataset metainfo + 评测 |
| 12 | 分布式 sampler 怎么分片？ | `DistributedSampler` 按 rank 切，epoch 打乱 |

---

## 相关笔记

- [[源码走读指南-BEVFormer]] — 14 天系统走读（本篇是速答版）
- [[BEVFormer详解]] — 原理与公式
- [[BEV感知全景]] — 三条路线对比
- [[开源项目索引]] — mmdet3d/OpenPCDet 入口
- [[Python手撕速答卡]] — 手撕默写
- [[项目深挖追问链]] — 项目追问
- [[从ADS跳槽-能力对照与查漏补缺]] — #10 开源栈差距
