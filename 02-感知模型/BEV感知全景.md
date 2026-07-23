---
tags: [overview, index]
created: "2026-07-21"
---

# BEV 感知全景

> BEV（Bird's Eye View）感知是当前自动驾驶感知的核心范式，将多视角 2D 图像特征转换到统一的鸟瞰图坐标系下进行感知。

---

## 为什么需要 BEV

- 多相机融合到统一坐标系
- 天然适合下游规划任务
- 时序信息更容易利用
- 目标不会发生遮挡和透视畸变

---

## 主流方法分类

### 1. 基于深度估计（Lift-Splat-Shoot）

**BEVDet** / **BEVDepth** — 显式预测深度分布，将 2D 特征投影到 3D → 拍平到 BEV

### 2. 基于 Transformer（BEVFormer 系）

**BEVFormer** — 使用可学习的 BEV queries，通过空间/时序 cross-attention 直接查询图像特征

### 3. 基于位置编码（PETR 系）

**PETR** — 3D 位置编码（3D PE），隐式完成视角变换，省去显式投影

### 4. 占据网络

**Occ3D / SurroundOcc / TPVFormer** — 从 BEV 升级到 3D 占据，处理不规则障碍物

---

## 关键模型速览

| 模型 | 年份 | 核心方法 | 代码 |
|------|------|----------|------|
| BEVDet | 2022 | Lift-Splat-Shoot | [GitHub](https://github.com/HuangJunJie2017/BEVDet) |
| BEVDepth | 2022 | 深度真值监督 | - |
| BEVFormer | 2022 | 时空 Transformer | [GitHub](https://github.com/fundamentalvision/BEVFormer) |
| BEVFormer v2 | 2023 | 透视监督 | - |
| PETR | 2022 | 3D 位置编码 | [GitHub](https://github.com/megvii-research/PETR) |
| PETRv2 | 2023 | 时序 PETR | - |
| StreamPETR | 2023 | 流式时序 | - |
| FB-BEV | 2024 | 前向-后向 BEV | - |

---

## 待读论文

- [ ] BEVFormer (ECCV 2022)
- [ ] BEVDet (2022)
- [ ] PETR (ECCV 2022)
- [ ] Occ3D (NeurIPS 2023)

---

## 相关笔记

- [[BEVFormer详解]] — Transformer 方式 BEV 特征变换
- [[BEVDet与BEVDepth]] — LSS 方式视图变换
- [[PETR系列]] — 3D 位置编码隐式变换
- [[占据网络与GOD]] — BEV → 3D 占据的升级
- [[COG协同占据栅格]] — 多车协同占据感知
- [[华为ADS技术方案]] — 🔥 华为量产方案：从 BEV 到 GOD 的演进
- [[端到端自动驾驶概览]]
