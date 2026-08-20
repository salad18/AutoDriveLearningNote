---
tags: [BEV, PETR, 3D-detection, position-encoding]
created: "2026-07-21"
updated: "2026-08-21"
---

# PETR 系列

> **论文**: PETR (ECCV 2022) → PETRv2 (ICCV 2023) → StreamPETR (ICCV 2023)
> 一句话导读：BEV 感知的第三条路线——**不给像素"量距离"（LSS），也不给 BEV 网格"问图像"（BEVFormer），而是直接给每个像素贴上"3D 地址"（3D 位置编码），让网络自己悟出"图像特征在 3D 空间的哪里"，从而省掉显式 BEV 特征图。**

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | PETR 的核心洞察（3D PE 替代显式投影）、3D PE 生成机制、PETRv2/StreamPETR 的时序演进、与 BEVFormer 的本质差异、代码走读、面试追问 |
| **需要的前置知识** | [[BEV感知全景]]（四路线）、[[BEVDet与BEVDepth]]（LSS 对照）、[[BEVFormer详解]]（Query 机制对照）、[[Transformer架构详解]]（位置编码） |
| **学完之后你能** | ① 说清"3D 位置编码如何隐式完成视角变换"；② 对比 PETR vs BEVFormer vs LSS 三条路线的本质差异；③ 解释 StreamPETR 为什么"存 Query 不存 BEV"；④ 理解位置编码类方法的优缺点 |
| **预计阅读时间** | 60-90 分钟 |

> [!tip] 怎么读这篇
> 这是 BEV 四篇的"对照收官篇"。**强烈建议先读 [[BEVDet与BEVDepth]] 和 [[BEVFormer详解]]**，带着"前两条路线怎么做的"来读，三条路线的差异会非常清晰。

---

## 〇、大白话总览

### PETR 在做什么？

输入 6 张相机图 → 输出 3D 检测框（车/人/锥桶在哪）。

它和前两条路线的区别只在一件事：**"图像特征怎么变成 3D 空间里的信息？"**

```
LSS 派:       先量距离（深度估计）→ 再把特征搬到 3D       → 显式，怕深度错
BEVFormer 派: 地面上铺网格（BEV Query）→ 投影到图像问答案 → 隐式，但要有 BEV 网格
PETR 派:      给每个像素贴上"3D 地址" → 网络自己悟        → 隐式，且没有 BEV 网格
```

### 核心思想（一句话）

**如果你能让每个图像特征知道自己在 3D 空间的精确位置，那就不需要构造显式的 BEV 网格——Transformer 可以隐式完成视角变换。**

> [!note] 打个比方
> 三条路线就像三种"找人"的方式：
> - LSS：先量出"他在几米外"，再按距离去找（量错了就找不到）。
> - BEVFormer：在地上画好格子，逐个格子问"这个格子里是谁"（格子多，问得慢）。
> - PETR：给每个"目击者"发一张 3D 地图，让他自己报"我在 3D 空间的哪、看到了什么"——**不需要地面网格，直接汇总目击者报告**。

---

## 一、核心思想（展开版）

PETR 的根本洞察：**如果你能让每个图像特征知道自己在 3D 空间的精确位置，那就不需要构造显式的 BEV 网格 — Transformer 可以隐式完成视角变换。**

```
BEVFormer:  Image → 显式 BEV grid (200×200) → 检测
PETR:       Image + 3D PE → Transformer Decoder → 检测 (无 BEV grid!)
```

> [!question] 追问自己：位置编码怎么就能替代"投影"？
> 因为**视角变换的本质是"知道每个像素对应 3D 空间哪里"**。
> - BEVFormer 的做法：BEV 网格有已知 3D 位置，投影到图像采样——**位置信息在"网格侧"**。
> - PETR 的做法：把 3D 位置直接编码进**图像特征本身**——位置信息在"特征侧"。Transformer 看到"这个特征带 3D 地址"，解码时自然知道该把它对应到 3D 的哪里。
> **位置信息从"网格侧"搬到"特征侧"，就是 PETR 的全部秘密。**

---

## 二、PETR v1 — 3D Position Encoding

### 2.1 3D PE 生成（逐步拆解）

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

**四步走明白：**

1. **生成 3D 锚点**：在自车周围空间铺一个 128×128×8 的 3D 网格（范围 ±51.2m × -5~3m），共 13 万个锚点——它们是"给 3D 空间打坐标"的参照物。
2. **投影锚点到相机**：每个 3D 锚点用相机内外参投影到图像上，落在图像内的锚点才有效（FOV mask，和 [[BEVFormer详解]] 的机制一样）。
3. **生成位置编码**：把锚点的 2D 投影坐标（pe_2d）和 3D 坐标（pe_3d）都做正弦编码，拼接后过 MLP，得到 256 维的"位置指纹"。
4. **加到图像特征**：每个有效像素的 2D 特征加上它对应 3D 位置的位置编码——**图像特征从此"知道"自己在 3D 的哪里**。

> [!note] 为什么同时用 2D 和 3D 编码？
> - **2D 坐标**（投影位置）：告诉特征"我在图像上的哪里"（图像内关系）。
> - **3D 坐标**（锚点位置）：告诉特征"我在 3D 空间的哪里"（跨视角关系）。
> 两者拼接 = **"图像位置 + 空间位置"双地址**。MLP 负责把它们融合成 Transformer 能用的形式。
> 关键：**位置编码只依赖相机内外参，不依赖图像内容**——所以可以预计算，和 BEVPoolv2 的"预计算索引"是同一类工程技巧（[[BEVDet与BEVDepth]] 提过）。

### 2.2 为什么不需要显式 BEV？（核心对比）

| | BEVFormer | PETR |
|---|---|---|
| **Query 类型** | BEV Queries（**局部**，每个对应固定网格位置） | Object Queries（**全局**，不绑定位置） |
| **空间定位方式** | Query 3D 位置 → 投影到图像采样 | 特征自带 3D PE，query 直接 attend |
| **显式 BEV 网格** | 需要（200×200） | **不需要** |
| **视图变换** | 显式投影 + Deformable Attention | 隐式（3D PE 在特征里，注意力自己学） |

> [!note] "全局 query"意味着什么？
> PETR 的 Object Query 和 DETR 一样是**可学习向量、不绑定空间位置**——每个 query 可以 attend 任何位置的图像特征。3D PE 保证了"空间对齐"：query 想知道"左前方 20 米有什么"，就能在**带着 3D 地址的特征**里找到对应信息。
> 而 BEVFormer 的 query 必须绑定网格位置（不然无法投影采样）——**这是"特征带地址"和"query 带位置"两种设计的根本分野**。

---

## 三、公式与直觉：3D PE 的数学形式

3D PE 采用经典的正弦位置编码（[[Transformer架构详解]] 里介绍过），对 3D 坐标逐维编码：

```
PE(x, y, z) = concat(
    [sin(x·ω_1), cos(x·ω_1), sin(x·ω_2), cos(x·ω_2), ...,
     sin(y·ω_1), cos(y·ω_1), ...,
     sin(z·ω_1), cos(z·ω_1), ...]
)
ω_i = 1 / 10000^(2i/d)   # 不同频率，从低频到高频
```

> [!note] 直觉：正弦编码在"编码什么"？
> - **不同频率 = 不同尺度的位置信息**：低频项编码"大致在哪"（粗粒度），高频项编码"精确在哪"（细粒度）。
> - 类似 GPS 坐标：先报"在北京"（低频），再报"海淀区中关村 XX 号"（高频）。
> - 3D PE 就是对 (x, y, z) 三个维度分别做这种编码，让 Transformer 能分辨"相距 1 米"和"相距 10 米"的两种空间关系。

**为什么不用"直接输入 3D 坐标数值"？**
> 直接输入 (20.3, 3.1, 1.0) 这种数值，网络很难区分"20.3"和"20.4"（数值差太小，被归一化后几乎无差异）。正弦编码把数值映射到**多频率周期信号**，任何微小位移都会在某个频率上产生明显差异——**编码比原始数值的信息密度高得多**。

---

## 四、PETRv2 → StreamPETR 演进（时序是 PETR 的主战场）

| | PETR v1 | PETR v2 | StreamPETR |
|---|---|---|---|
| **时序** | 无 | 1 帧 temporal PE | **Memory Queue (8帧)** |
| **时序机制** | — | ego-motion aligned PE | Recurrent query propagation |
| **NDS** | 50.4 | 58.2 | **63.7** |
| **Memory** | 0 | ~10MB (1 BEV frame) | ~0.23MB (8×900 queries) |

**StreamPETR 的关键**: Memory Queue 存 Object Queries 而非 BEV features → 轻量 → 可以追更长时序。

> [!note] 三条演进逻辑（每个版本在补什么短板？）
> 1. **PETRv2**：v1 没有时序（单帧），v2 把上一帧的特征按 ego-motion 对齐后加进时序编码——**补上"过去"**。
> 2. **StreamPETR**：v2 存的是"一帧 BEV 特征"（10MB），StreamPETR 发现**与其存整帧特征，不如存"上一帧的 900 个 Object Query"**（0.23MB）——把"状态"从"空间快照"变成"Query 记忆"，又轻又能流式更新。
> 3. **NDS 50.4 → 58.2 → 63.7**：时序带来的提升巨大（+13 个点），说明**对 PETR 这种全局 query 结构，时序是性能放大器**。

> [!warning] 一个关键的架构观察
> 注意 NDS 63.7 已经超过 BEVFormer-B（56.9）——**PETR 系在 2023 年实现了对 BEVFormer 的全面反超**。原因：① 没有显式 BEV 网格，省下算力给更大的模型；② 时序方案（Query 记忆）比"存 BEV 帧"更高效。**这就是"第三条路线"的价值——它找到了更高效的时序形态。**

---

## 五、代码走读：3D PE 的实现要点

```python
# 简化自 PETR 官方实现 (projects/mmdet3d_plugin/models/dense_heads/petr_head.py)

class PETRPositionalEncoding(nn.Module):
    """3D 位置编码：为图像特征注入 3D 空间地址"""
    def __init__(self, num_feats=128, num_anchor=128, num_z=8):
        super().__init__()
        # 3D 锚点网格（ego 坐标系）
        x = torch.linspace(-51.2, 51.2, num_anchor)
        y = torch.linspace(-51.2, 51.2, num_anchor)
        z = torch.linspace(-5, 3, num_z)
        self.anchor_3d = torch.stack(torch.meshgrid(x, y, z), dim=-1)  # [128,128,8,3]

    def forward(self, img_feats, cam_params):
        """
        img_feats: list of [B, C, H, W] (各相机特征)
        cam_params: 各相机内参 K + 外参 RT
        """
        B, C, H, W = img_feats[0].shape
        out_feats = []
        for cam_idx, feat in enumerate(img_feats):
            # 1. 3D 锚点投影到该相机
            pts_2d = project(self.anchor_3d, cam_params[cam_idx])  # [..., 2]
            valid = inside_image(pts_2d, H, W)                      # FOV mask

            # 2. 2D + 3D 双坐标正弦编码 → MLP 融合
            pe = self.sinusoidal(pts_2d[valid])          # 2D 位置编码
            pe = torch.cat([pe, self.sinusoidal_3d(
                self.anchor_3d[valid])], dim=-1)         # + 3D 位置编码
            petr_pe = self.pe_mlp(pe)                    # [N_valid, 256]

            # 3. 加到图像特征（只有有效像素加）
            feat = feat.clone()
            feat[valid] = feat[valid] + petr_pe
            out_feats.append(feat)
        return out_feats  # 图像特征从此"自带 3D 地址"
```

> [!tip] 读代码要点
> 核心就一个操作：**`feat[valid] = feat[valid] + petr_pe`**——"把 3D 地址加到特征上"。后面的 Transformer Decoder 完全不需要知道相机内外参，**因为它吃到的特征已经"带地址"了**。这是 PETR 结构简洁的根源：几何信息全部前移到编码阶段，解码阶段保持纯净。

---

## 六、面试官追问

### Q1: anchor points 128×128×8 = 131K 个，密度够吗？

对 51.2m 范围: 每 0.8m 一个 anchor。物体大小 ≥ 2m（car）→ 被 ≥ 2-3 个 anchor 覆盖 → 够用。小物体（pedestrian 0.5m）可能只对应 1 个 anchor → 不充分。消融: 256×256 → NDS +0.8（小物体改善）。

### Q2: StreamPETR 的 memory queue 会不会积累错误？

会（error accumulation）。缓解：高 confidence threshold 入 queue + detach memory（不反向传播）。实验中 8 帧有边际收益 → 更长可能有害。

### Q3: PETR 相比 BEVFormer 的优缺点是什么？（高频！）

| | PETR | BEVFormer |
|---|---|---|
| **优点** | 结构简洁（无 BEV 网格）、时序高效（Query 记忆）、NDS 更高 | 显式网格可解释、Deformable 采样计算可控 |
| **缺点** | 3D PE 设计空间大（调参敏感）、全局注意力显存大、无显式 BEV 特征（下游任务要另想办法） | 需要显式 BEV 网格、时序存整帧（10MB）、速度慢 |

**回答框架**：先说"核心差异在位置信息放哪侧"，再说各自优缺点，最后落到"为什么 PETR 系后期反超"。

### Q4: 既然 PETR 更强，为什么工业界还常用 BEVFormer 结构？

三个原因：
1. **下游任务**：BEV 特征可以直接喂给分割/占用/端到端模块（[[端到端自动驾驶概览]] 的链路），PETR 没有显式 BEV 特征，下游要重建。
2. **工程成熟度**：BEVFormer 系开源生态更完整（mmdet3d 支持好），部署方案成熟。
3. **速度可控**：Deformable Attention 的采样数可调，显存可控；PETR 全局注意力的显存随分辨率平方增长。

---

## 七、常见误区速查

> [!warning] 新手最常踩的坑
> 1. **以为 PETR "没有视图变换"**——不是！视图变换被**隐式化**了：3D PE 把几何信息编码进特征，Transformer 在解码时隐式完成变换。变换仍然存在，只是不显式。
> 2. **以为 3D PE 是"直接输入 3D 坐标"**——是正弦编码 + MLP 融合，不是裸坐标（裸坐标信息密度太低）。
> 3. **混淆"特征带地址"和"query 带位置"**——BEVFormer 是 query 带位置（投影采样），PETR 是特征带地址（编码进特征）。这是两条路线最本质的分野。
> 4. **以为 PETR 不需要相机内外参**——需要！内外参用于"生成 3D PE 时的锚点投影"（2.1 的 Step 2）。只是解码阶段不再用。
> 5. **忽略时序**——PETR 的 NDS 从 50.4（无时序）到 63.7（StreamPETR），时序是它最大的性能杠杆。

---

## ✅ 检验自己（自测题）

> [!question] Q1：用你自己的话解释"3D 位置编码如何替代显式投影"。
> 提示：位置信息放在"特征侧"还是"网格侧"？

> [!success]- 参考答案
> 视角变换的本质是"知道每个像素对应 3D 空间哪里"。PETR 把 3D 位置编码直接加到图像特征上（特征侧），让特征"自带 3D 地址"；Transformer Decoder 解码时自然知道该把特征对应到 3D 的哪里，无需显式构造 BEV 网格或做投影采样。对比 BEVFormer 把位置放在 query 侧（网格投影到图像采样）——位置信息从"网格侧"搬到"特征侧"就是 PETR 的核心。

> [!question] Q2：3D PE 生成的四步是什么？为什么同时用 2D 和 3D 坐标编码？
> 提示：锚点 → 投影 → 编码 → 相加。

> [!success]- 参考答案
> ① 在 ego 坐标系生成 3D 锚点网格（128×128×8）；② 用内外参把锚点投影到各相机，FOV mask 过滤有效点；③ 对 2D 投影坐标和 3D 锚点坐标分别做正弦编码并拼接，过 MLP 融合成位置向量；④ 把位置向量加到对应像素的图像特征上。同时用 2D 和 3D：2D 编码"图像位置"（图像内关系），3D 编码"空间位置"（跨视角关系），双地址让特征既懂图像又懂空间。

> [!question] Q3：PETR 和 BEVFormer 的"位置信息"分别放在哪一侧？各自的代价是什么？
> 提示：特征侧 vs query 侧。

> [!success]- 参考答案
> PETR 放在特征侧（3D PE 编码进图像特征），代价是 3D PE 的设计空间大、调参敏感，且没有显式 BEV 特征供下游任务使用。BEVFormer 放在 query 侧（BEV query 投影到图像采样），代价是需要显式 BEV 网格、时序要存整帧 BEV（10MB）、全局注意力慢。本质差异：一个"特征带地址"，一个"query 带位置"。

> [!question] Q4：StreamPETR 为什么"存 Query 不存 BEV"？好处是什么？
> 提示：0.23MB vs 10MB。

> [!success]- 参考答案
> 时序信息本质是"上一帧的物体状态"，用 900 个 Object Query 就能承载（0.23MB），比存整帧 BEV 特征（10MB）轻 40 倍以上。Query 记忆还能流式传播（recurrent propagation），支持追更长时序。轻量 + 长效 = NDS 63.7 的关键。

> [!question] Q5：既然 PETR 系 NDS 更高，工业界为什么还常用 BEVFormer 结构？给出至少两个理由。
> 提示：下游任务、工程生态。

> [!success]- 参考答案
> ① 下游任务：BEV 特征可直接喂分割/占用/端到端模块，PETR 无显式 BEV 特征，下游要重建；② 工程成熟度：BEVFormer 系开源生态完善、部署方案成熟；③ 显存可控：Deformable 采样数可调，PETR 全局注意力显存随分辨率平方增长。所以"论文最强"和"工程最爱"经常不是同一个。

---

## 🛠 动手练习

### 练习 1：手算 3D PE 的"投影 → 编码 → 相加"（20 分钟）

给定一个 3D 锚点 (10, 3, 1)（前方 10 米、偏右 3 米、高 1 米），相机内参 `K = [ [1000, 0, 640], [0, 1000, 360], [0, 0, 1] ]`，外参为单位变换：
1. 用 [[3D视觉与投影几何]] 的投影公式算像素坐标 (u, v)。
2. 假设图像 1280×720，判断该锚点是否有效（FOV）。
3. 思考：如果锚点投影到图像外（比如在相机正后方），这个位置编码还会被加到特征上吗？（答案：不会，被 FOV mask 过滤）

> [!tip] 做完后自问
> ① 锚点在图像外的像素会拿到"3D 地址"吗？② 6 个相机视角下，同一个 3D 锚点可能在几个相机里有效？（多个——这就是"跨视角一致"的来源。）

### 练习 2：读 PETR 官方代码（60-90 分钟）

打开 GitHub: megvii-research/PETR，读 `projects/mmdet3d_plugin/models/dense_heads/petr_head.py`：
1. 找到 3D PE 生成代码（`position_embedding` 相关），对照本文 2.1 的伪代码。
2. 找到 `positional_encoding` 的实现，确认它是不是正弦编码。
3. 回答：`num_anchor=128` 时，3D PE 的总维度是多少？（提示：128×3 坐标 × 2(sin/cos) → 拼接 2D 后过 MLP → 256）

> [!tip] 读代码的姿势
> 先找"形状"：位置编码输出的形状必须是 [N, 256] 才能和图像特征相加。顺着形状找实现，比逐行读快得多。

### 练习 3：跑通 PETR 推理（可选，1-2 天）

按官方仓库在 nuScenes mini 上跑通推理，然后：
1. 可视化检测结果，对比 [[BEVFormer详解]] 练习里跑的结果。
2. 修改 `num_anchor` 从 128 → 256，看 NDS 变化（论文说 +0.8）。
3. 回答：为什么小物体改善最大？

> [!warning] 环境提示
> PETR 代码基于 mmdet3d，环境配置和 BEVFormer 一样是"版本地狱"。跑不通先看 [[训练排错实战手册]]，报错记进 [[2026-08-20]]。

---

## ➡️ 下一步学什么

按知识库学习路径，读完本篇你应该接着：

1. **[[占据网络与GOD]]** —— BEV 四篇收官后进入 3D 占据（如果还没读）。
2. **[[VAD详解]]** —— 端到端对照：矢量化表征如何让规划快 3 倍。
3. **[[Transformer进阶知识]]** —— 想深入位置编码（RoPE 等）时看。
4. **[[华为ADS技术方案]]** —— 看量产方案怎么把感知路线落地（GOD 大网）。

> 💡 至此 BEV 感知四条路线全部学完（LSS / BEVFormer / PETR / 占据）。建议停下来做一个"四路线对比表"（路线 / 视图变换 / 深度处理 / 时序 / 优缺点），这个表会是你面试和读论文的利器。

---

## 相关笔记

- [[BEV感知全景]] — BEV 四路线总览
- [[BEVDet与BEVDepth]] — LSS 派对照
- [[BEVFormer详解]] — Transformer 派对照
- [[占据网络与GOD]] — 第四条路线（3D 占据）
- [[Transformer架构详解]] — 位置编码基础
- [[Transformer进阶知识]] — RoPE 等进阶编码
