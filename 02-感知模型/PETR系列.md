---
tags: [BEV, PETR, 3D-detection, position-encoding]
created: "2026-07-21"
updated: "2026-07-28"
---

# PETR 系列

> **论文**: PETR (ECCV 2022) → PETRv2 (ICCV 2023) → StreamPETR (ICCV 2023)
> 核心路线：**通过 3D Position Encoding 将 2D 图像特征"位置化"，再用 Transformer Decoder 直接输出 3D 检测 — 不需要显式 BEV 特征图。**

---

## 一、核心思想

PETR 的根本洞察：**如果你能让每个图像特征知道自己在 3D 空间的精确位置，那就不需要构造显式的 BEV 网格 — Transformer 可以隐式完成视角变换。**

```
BEVFormer:  Image → 显式 BEV grid (200×200) → 检测
PETR:       Image + 3D PE → Transformer Decoder → 检测 (无 BEV grid!)
```

---

## 二、PETR v1 — 3D Position Encoding

### 2.1 3D PE 生成

```python
# Step 1: 在 ego 坐标系中定义 3D anchor points (meshgrid)
x = torch.linspace(-51.2, 51.2, 128)  # 128 points in x
y = torch.linspace(-51.2, 51.2, 128)  # 128 points in y
z = torch.linspace(-5, 3, 8)          # 8 points in z
anchor_3d = torch.stack(torch.meshgrid(x, y, z), dim=-1)
# [128, 128, 8, 3] — total: 131072 个 anchor points

# Step 2: 将每个 anchor point 投影到每个相机
for cam in range(6):
    pts_2d = project_3d_to_2d(anchor_3d, cam_K, cam_RT)
    valid = (pts_2d[...,0]>=0) & (pts_2d[...,0]<W) & (pts_2d[...,1]>=0) & (pts_2d[...,1]<H)
    
    # Step 3: 位置编码
    pe_2d = positional_encoding(pts_2d[valid])  # sinusoidal
    pe_3d = positional_encoding(anchor_3d[valid])
    petr_pe = MLP(concat(pe_2d, pe_3d))  # [N_valid, 256]
    
    # Step 4: 加到图像特征上
    img_feat[cam][valid_positions] += petr_pe
```

### 2.2 为什么不需要显式 BEV？

PETR 的 Object Queries 是**全局的** — 每个 Query 可以直接 attend 到任何位置的图像特征。3D PE 保证了空间对齐。

而 BEVFormer 的 BEV Queries 是**局部的** — 每个 Query 对应固定 BEV 位置 → 必须通过相机投影来"定位"。

---

## 三、PETRv2 → StreamPETR 演进

| | PETR v1 | PETR v2 | StreamPETR |
|---|---|---|---|
| **时序** | 无 | 1 帧 temporal PE | **Memory Queue (8帧)** |
| **时序机制** | — | ego-motion aligned PE | Recurrent query propagation |
| **NDS** | 50.4 | 58.2 | **63.7** |
| **Memory** | 0 | ~10MB (1 BEV frame) | ~0.23MB (8×900 queries) |

**StreamPETR 的关键**: Memory Queue 存 Object Queries 而非 BEV features → 轻量 → 可以追更长时序。

---

## 四、面试官追问

### Q: anchor points 128×128×8 = 131K 个，密度够吗？

对 51.2m 范围: 每 0.8m 一个 anchor。物体大小 ≥ 2m（car）→ 被 ≥ 2-3 个 anchor 覆盖 → 够用。小物体（pedestrian 0.5m）可能只对应 1 个 anchor → 不充分。消融: 256×256 → NDS +0.8（小物体改善）。

### Q: StreamPETR 的 memory queue 会不会积累错误？

会（error accumulation）。缓解：高 confidence threshold 入 queue + detach memory（不反向传播）。实验中 8 帧有边际收益 → 更长可能有害。

---

> 📚 **相关**: [[BEVFormer详解]], [[BEVDet与BEVDepth]], [[Transformer进阶知识]]
