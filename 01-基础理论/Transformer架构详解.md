---
tags: [foundation, transformer, attention]
created: "2026-07-21"
updated: "2026-08-21"
---

# Transformer 架构详解

> 一句话导读：Transformer 的核心只有一件事——**用"注意力"让序列里的每个元素互相交换信息**。本篇从 Q/K/V 的直觉讲起，逐步推导到多头注意力、交叉注意力，最后落到它在 BEV 感知里的实际用法。

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | Self-Attention 的完整推导与直觉、Q/K/V 到底是什么、Multi-Head / Cross-Attention / Deformable Attention、位置编码、ViT / DETR / Swin 三个经典变体、BEVFormer 中的实际应用 |
| **需要的前置知识** | [[模型架构演进]]（⭐ 架构演进地图，衔接 MLP/CNN→Transformer）、[[计算机视觉基础]]（CNN 部分）；会看矩阵乘法即可，本篇每个公式都配白话解释 |
| **学完之后你能** | ① 用自己的话讲清楚 Q/K/V 和注意力公式；② 说清 Self vs Cross Attention 的区别；③ 理解 Object Query 机制；④ 为读懂 BEVFormer、PETR、UniAD 扫清障碍 |
| **预计阅读时间** | 60-90 分钟 |

> [!tip] 为什么这篇是"地基中的地基"
> 现在 BEV 感知（BEVFormer）、端到端（UniAD）、世界模型（OccWorld）几乎全部建立在 Transformer 上。**Attention Is All You Need 是 2017 年的论文，但今天所有自动驾驶感知模型都在用它的思想。** 这篇读透，后面所有模型笔记都轻松一半。

---

## 一、为什么 Transformer 重要

在 BEV 感知和端到端自动驾驶中，Transformer 无处不在：

| 应用 | 模型 | Transformer 的角色 |
|------|------|-------------------|
| BEV 特征变换 | BEVFormer | 时空 Cross-Attention |
| 3D 检测 | DETR3D | Object Query 机制 |
| 位置编码 | PETR | 3D 位置编码 + Transformer Decoder |
| 端到端规划 | UniAD | 多任务 Query 交互 |
| 世界模型 | OccWorld | GPT 式生成未来占据 |

**那 CNN 不够吗？为什么要换 Transformer？**

> [!note] CNN vs Transformer 的核心差异（直觉）
> - **CNN**：局部扫描。每个输出像素只看输入的一个小窗口（感受野有限），要靠堆很多层才能"看全局"。就像近视眼，只能一点一点看。
> - **Transformer**：全局注意力。每个输出 token **一步直接看到所有输入 token**，并自动决定"该重点看谁"。就像扫一眼全场，立刻知道重点在哪。
>
> 代价：全局注意力计算量是 O(N²)（N 是 token 数），比 CNN 大得多——这正是后面 Deformable Attention 出现的原因。

---

## 二、核心机制

### 2.1 Self-Attention（自注意力）——逐层拆解

#### 第一步：什么是 token 和序列

Transformer 处理的是"序列"。在 NLP 里 token 是词；在视觉里 token 可以是**图片切成的 Patch**、**BEV 网格上的格子**、**DETR 的 Object Query**。

假设输入是 3 个 token，每个 token 用 2 维向量表示：

```
X = [ x1 ]   x1 = [0.5, 0.1]   （"猫"）
    [ x2 ]   x2 = [0.2, 0.8]   （"坐"）
    [ x3 ]   x3 = [0.9, 0.3]   （"垫子"）
```

#### 第二步：Q / K / V 到底是什么

**每个 token 通过三个不同的线性变换（三个矩阵 W_Q, W_K, W_V），生成三个"角色"向量：**

```
Q = X·W_Q    Query（查询）："我要找什么信息？"    —— 我在问问题
K = X·W_K    Key  （键）：  "我能匹配什么查询？"    —— 我的索引标签
V = X·W_V    Value（值）：  "我实际提供什么内容？"  —— 我的答案内容
```

> [!note] 查词典的比喻（最经典的直觉）
> 你要在词典里查"attention"这个词：
> - **Query** = 你要查的那个词（"attention"）
> - **Key** = 词典里所有词条的词头（每个词条的索引标签）
> - **Value** = 每个词条的解释内容
>
> 查的过程 = 把你的 Query 和所有 Key 比较相似度 → 找到最相关的词条 → 按相关度加权汇总它们的 Value（解释）。这就是注意力的一步。

#### 第三步：注意力公式（一个数字一个数字地看）

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

**拆开看：**

1. **`Q·Kᵀ`**：计算"每个 token 和每个 token 的相关度"。得到一个 N×N 的矩阵，第 i 行第 j 列 = token i 的 Query 和 token j 的 Key 的点积（点积越大 = 方向越一致 = 越相关）。

2. **`/ √d_k`**：除以 Key 向量维度的平方根。
   > 为什么？向量维度大时点积数值会变大，导致 softmax 输出接近 one-hot（极端化），梯度极小、训练困难。除以 √d_k 把数值拉回合理范围，softmax 曲线更平滑，梯度更健康。**这不是可选项，是保训练稳定的关键。**

3. **`softmax(...)`**：按行做归一化，把每行的相关度变成"概率"（和为 1）。含义：token i 应该把注意力按多少比例分给每个 token j。

4. **`· V`**：用归一化后的权重，对所有 Value 做加权求和，得到每个 token 的新表示。
   > 结果：**每个 token 的新向量 = 全序列信息的加权汇总**，权重由"相关度"决定。这就是"自注意力"——序列内部自己互相更新。

#### 数字小例子（感受一下）

```
Q·Kᵀ 得到相关度矩阵（未缩放）:
         token1  token2  token3
token1: [  2.0    0.5    1.2 ]
token2: [  0.5    1.8    0.3 ]
token3: [  1.2    0.3    2.5 ]

softmax 按行归一化（假设已缩放）:
token1: [  0.60   0.15   0.25 ]   ← token1 把 60% 注意力给"自己"，25% 给 token3
token2: [  0.25   0.60   0.15 ]
token3: [  0.25   0.15   0.60 ]

token1 的新向量 = 0.60·V1 + 0.15·V2 + 0.25·V3
```

> [!note] 为什么"自己"权重往往最高？
> 因为 token 和它自己天然最相关（点积最大）。这在视觉里很常见——注意力图经常在对角线上最亮。多轮叠加后，信息逐步在序列里传播。

#### 关键认识

- **Self-Attention = Q、K、V 都来自同一个序列**。序列内部互相更新。
- 一次 Self-Attention 让每个 token **一步看到全局**（感受野 = 整个序列），这是 CNN 堆几十层都做不到的。
- 但代价是 O(N²)：N 个 token 要算 N×N 的相关度矩阵。**N 越大越贵**（后面 Deformable Attention 就是为了省钱）。

### 2.2 Multi-Head Attention（多头注意力）——"多个视角同时看"

#### 为什么需要多头？

一个注意力头只能学一种"相关模式"。但"相关"有很多种：位置上的相关（相邻的词）、语法上的相关（主谓）、语义上的相关（近义词）、视觉里的几何相关（同一辆车）……

**多头 = 把 Q/K/V 切分成 h 份，每个头独立算注意力，最后拼接。**

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

> [!note] 直觉
> 就像开评审会：不是一个人看完全场，而是派 h 个专家（头），每个专家专注看一类关系（位置、语义、几何……），最后把各专家的结论汇总（W_O 投影拼接结果）。**每个头负责一类"关系"，合起来才完整。**

典型配置：`d_model=256, h=8, d_k=32` → 每个头只在 32 维子空间里算注意力，8 个头拼回 256 维。**总计算量和单头 256 维差不多，但表达能力大幅提升。**

> [!tip] 视觉里有个著名现象
> 在 ViT 的可视化中，有的头专门学"位置关系"（对角线结构），有的头专门学"语义关系"（同类物体互相吸引）。这就是多头分工的直观证据。

### 2.3 Cross-Attention（交叉注意力）——两个序列的"对话"

**Self-Attention：Q、K、V 来自同一序列（自己跟自己聊）。**
**Cross-Attention：Q 来自一个序列，K、V 来自另一个序列（两个序列对话）。**

```
Q 来自序列 A（比如 BEV Query / Object Query）
K, V 来自序列 B（比如图像特征）
```

**BEVFormer 中的用法（提前剧透，详细见 [[BEVFormer详解]]）：**

1. BEV 网格上每个格子是一个 **BEV Query**（Q 序列）。
2. 它想知道："我这个格子在地面上，对应的图像区域长什么样？"
3. 通过相机内外参，把 BEV 格子的 3D 位置投影到各相机的 2D 图像上，采样得到 K、V（图像特征）。
4. Cross-Attention 让 BEV Query **从图像里"提取"它需要的视觉信息**，填充自己的特征。

> [!note] 一句话总结
> **Cross-Attention 是"提问者"从"回答者"那里取信息的机制。** 在 BEV 里，BEV 网格是提问者，图像是回答者。在 DETR 里，Object Query 是提问者，整张图是回答者。**整个自动驾驶感知的 Transformer 化，本质就是把"提问-回答"机制用到不同的空间。**

### 2.4 Deformable Attention（可变形注意力）——让注意力"偷懒但精准"

#### 问题

标准 Attention 是 O(N²)：每个 BEV query 要和**所有**图像特征位置算注意力。BEV 网格 200×200 = 40000 个 query，图像特征 50000 个位置——40000 × 50000 的矩阵，显存直接爆炸。

#### 解法

**Deformable Attention 只采样 K 个关键点**（K 通常 4-8）：

```
DeformAttn(Q, p, x) = Σ_k  W_k · x(p + Δp_k)
```

- `p`：参考点（比如 BEV 格子投影到图像上的位置）
- `Δp_k`：网络学出来的偏移量（"别死盯着投影点，往周围挪一点可能更准"）
- 只在这 K 个偏移点附近采样特征

> [!note] 直觉
> 标准注意力是"全场挨个看"（O(N²)），可变形注意力是"我知道重点在哪几个位置，只去看那几个"（O(N×K)）。就像高手开车只看几个关键点（前车尾灯、后视镜、路口信号），不看整幅画面。
>
> **为什么偏移是可学习的？** 因为"投影点"不一定最准——透视投影、标定误差、物体运动都会让正确采样点偏离投影点。让网络自己学偏移，就是在学"投影不准的部分怎么补偿"。

**在 BEV 中的应用**：BEVFormer 的 Spatial Cross-Attention 用的就是 Deformable Attention——每个 BEV query 只在投影位置周围采 4 个点，而不是看整张图。这是 BEVFormer 能跑起来的**工程关键**。

### 2.5 Position Encoding（位置编码）——给 Transformer 装上"空间感"

#### 为什么需要？

**Attention 本身对位置一无所知**：如果把 token 顺序打乱，注意力计算结果完全一样（它是"集合"操作，不是"序列"操作）。但"猫坐在垫子上"和"垫子坐在猫上"意思完全不同；BEV 里"左前 20 米"和"右后 20 米"也完全不同。**必须把位置信息注入。**

| 类型 | 方法 | 使用场景 |
|------|------|----------|
| **正弦位置编码** | sin/cos 函数生成固定位置向量 | 原始 Transformer（NLP） |
| **可学习位置编码** | 直接学一组位置参数 | ViT、BEVFormer |
| **3D 位置编码** | 3D 坐标 → MLP → 位置向量 | PETR（核心创新！） |
| **相机参数编码** | 内外参 → 编码 | BEV 特征变换 |

> [!note] 三种理解层次
> 1. **为什么要有**：Attention 无位置感，必须显式注入。
> 2. **怎么做**：正弦编码（数学公式生成）或可学习编码（训练学出来），加到 token 向量上（`x + pos`）。
> 3. **BEV 的特殊性**：BEV 网格有明确的物理坐标（x 米, y 米），所以位置编码可以直接用 **3D 物理坐标**——这就是 PETR 的核心思想（[[PETR系列]] 里细讲）：**用 3D 坐标做位置编码，让网络隐式学会"图像像素在 3D 空间的哪里"**，省掉显式投影。

---

## 三、经典变体（必须认识的三位）

### ViT（Vision Transformer）——把图片当句子读

```
图像 → 切分成 16×16 Patches → Patch Embedding → Transformer Encoder → 分类
```

- **核心思想**：图片 = 一串"像素块单词"。224×224 图切成 14×14=196 个 16×16 patch，每个 patch 展平成一个 token，加上位置编码，扔进标准 Transformer。
- **为什么要认识它**：BEVFormer 等模型的特征提取沿用了 ViT 的"patch 化 + 注意力"思想。
- **局限性**：纯 Transformer 需要海量数据预训练（JFT-300M 那种量级），数据少时不如 CNN 好用——这也是 Swin 这种"混合派"出现的原因。

### DETR（DEtection TRansformer）——检测界的范式革命

```
图像 → CNN Backbone → Transformer Encoder → Object Queries → Transformer Decoder → 预测框
```

- **Object Query**：固定 100 个可学习向量。每个 query 通过 Decoder 的 Cross-Attention **反复"询问"图片特征**，最终"长成"一个检测结果（框 + 类别）。
  > 直觉：100 个"探员"，每个探员负责盯一类目标（有的盯车、有的盯人），反复看图后汇报自己盯到了什么。
- **二分图匹配（Hungarian 匹配）**：训练时把 100 个预测和真值做**全局最优配对**（一对多、多对一都不允许），直接端到端训练。
- **革命性意义**：**去掉了 anchor 设计和 NMS 后处理**——以前检测 pipeline 里最麻烦的两个手工模块，被"可学习 + 匹配"替代。
- **自动驾驶的意义**：DETR3D 把 Object Query 从 2D 扩展到 3D（query 带 3D 坐标，投影到图像采样），PETR 更进一步（3D 位置编码），UniAD 用 query 串联起检测→跟踪→规划。**整条技术线都源于 DETR。**

### Swin Transformer——"局部注意力 + 窗口移位"

- 问题：全局注意力太贵，而且**不保留图像的多尺度结构**（CNN 天然有金字塔，ViT 没有）。
- 解法：
  1. **窗口注意力**：只在 7×7 的小窗口内做 Self-Attention（局部，便宜）。
  2. **移位窗口**：下一层窗口整体偏移，让不同窗口的信息能跨窗口流动（全局，靠多层累积）。
  3. **层次化结构**：像 CNN 一样逐层降采样（4× → 8× → 16×），天然适配 FPN 多尺度检测。
- 定位：**Transformer 版的高性能 Backbone**，BEVFormer v2 等模型用它替换 ResNet。

---

## 四、BEV 感知中的 Transformer——BEVFormer 的时空注意力

### 4.1 整体数据流

```
            ┌─────────────────────┐
            │   Temporal          │
BEV Queries │   Self-Attention    │ ← 历史 BEV 特征
    ────→   │   (时序融合)        │
            └─────────┬───────────┘
                      │
            ┌─────────▼───────────┐
            │   Spatial           │
            │   Cross-Attention   │ ← 当前多相机图像特征
            │   (空间融合)        │
            └─────────┬───────────┘
                      ▼
                更新后的 BEV 特征
```

### 4.2 三个关键设计（逐条解释）

**① BEV Query = 可学习的位置锚点，分布在鸟瞰网格上**

- BEV 空间被分成 200×200 的网格，每个格子 = 0.4m×0.4m 的真实地面区域。
- 每个格子有一个可学习向量（query），它代表"我这个格子里有什么"。
- 直觉：**把地面铺一张"待填写的表格"，每个格子等着从图像里收集证据。**

**② 参考点投影：BEV 格子 → 3D 参考点 → 相机投影 → 2D 采样位置**

- 每个 BEV query 有已知的地面位置 (x, y)（比如 x=20m, y=3m 表示"前方 20 米偏右 3 米"）。
- 给定一个预设高度 z（比如 1m），得到 3D 参考点 (20, 3, 1)。
- 用相机内外参把该 3D 点投影到每个相机的 2D 图像上，得到采样点 (u, v)。
- 在这个采样点附近做 Deformable Attention 采特征。

> [!note] 这一步为什么是"后向投影"？
> 方向是 **3D → 2D**（从 BEV 网格出发，投影到图像找证据）。和 LSS 的"2D → 3D"（图像像素抬升到 3D）正好相反。两条路线谁优谁劣，[[3D视觉与投影几何]] 和 [[BEVDet与BEVDepth]] 会细讲。

**③ 时序对齐：历史 BEV 特征根据 ego 运动对齐到当前坐标系**

- 上一帧的 BEV 特征里，物体位置是**上一帧车的位置**定义的。
- 这一帧车往前开了 2 米，上一帧的"前方 20 米"现在其实在"前方 18 米"。
- 用 ego-motion（自车位姿变化）做坐标变换，把历史 BEV 特征**搬到当前坐标系**，再和当前 BEV query 做 Self-Attention 融合。
- 结果：**静止物体越看越清楚，动态物体也能借历史信息补全遮挡。**

> [!tip] 为什么时序这么重要
> 单帧图像可能有遮挡、模糊；多帧叠加后，被挡住的物体可能在某帧露出来。BEVFormer 的时序注意力让"记忆"参与感知——这也是它比单帧方案强的主要原因之一。

---

## 五、关键概念速记

| 概念 | 含义 | 一句话记忆 |
|------|------|-----------|
| **Q/K/V** | Query/Key/Value 三元组 | 问什么 / 配什么 / 答什么 |
| **Self-Attention** | Q、K、V 来自相同序列 | 自己跟自己对话 |
| **Cross-Attention** | Q 和 K/V 来自不同源 | 提问者向回答者取信息 |
| **Multi-Head** | 多头并行看不同关系 | 多个专家分工评审 |
| **Deformable Attn** | 稀疏采样 K 个关键点 | 只看重点，O(N×K) 而非 O(N²) |
| **√d_k 缩放** | 防止点积过大 | 保护 softmax 梯度 |
| **FFN** | Feed-Forward Network，位置独立 MLP | 每个 token 独立"过脑"一次 |
| **LayerNorm** | 层归一化，稳定训练 | 稳定每层输出分布 |
| **Object Query** | DETR 系的可学习检测锚点 | 100 个"探员" |
| **Position Encoding** | 注入位置信息 | 给 Attention 装空间感 |

> [!note] 为什么 Transformer 块是"Attention + FFN"的组合？
> Attention 负责**token 之间交换信息**（横向），FFN 负责**每个 token 自己加工信息**（纵向）。两者交替堆叠：先交流、再思考、再交流、再思考……这就是 Transformer Encoder 的"一课一课"。

---

## 六、常见误区速查

> [!warning] 新手最常踩的坑
> 1. **以为 Q/K/V 是三个"输入"**——不是，它们是从同一个输入 X 用三个矩阵算出来的三个视角。
> 2. **以为注意力权重是"学出来的"**——权重是**输入算出来的**（softmax(QKᵀ/√d)），学的是 W_Q/W_K/W_V 这些投影矩阵。
> 3. **混淆 Self 和 Cross**——看 Q 和 K/V 是否同源，同源是 Self，异源是 Cross。
> 4. **以为 Deformable Attention 只是"工程加速"**——它同时是**归纳偏置**：强制模型关注局部邻域，反而更贴合图像的自然结构，效果往往不比全局差。
> 5. **忘记位置编码**——没有位置编码，BEV 网格就退化成"无空间感的集合"，检测全乱。

---

## ✅ 检验自己（自测题）

> [!question] Q1：用自己的话解释 Q/K/V 分别是什么，并用"查词典"比喻讲一遍 Attention 的流程。
> 提示：想清楚"问什么 / 配什么 / 答什么"三个角色。

> [!success]- 参考答案
> Q（Query）是当前 token 想问的问题，K（Key）是所有 token 的"索引标签"，V（Value）是所有 token 的"实际内容"。流程：拿 Q 和所有 K 做点积算相关度 → 除以 √d_k 并 softmax 归一化成注意力权重 → 用权重对所有 V 加权求和。类比查词典：Query=要查的词，Key=词条词头，Value=词条解释，加权求和=综合多本词典的答案。

> [!question] Q2：公式里的 `√d_k` 是干什么的？去掉会怎样？
> 提示：从"数值范围 → softmax 梯度"这条链想。

> [!success]- 参考答案
> d_k 大时点积数值大，softmax 输出趋于 one-hot（两极分化），对应位置梯度趋近 0，训练困难。除以 √d_k 把点积拉回合理范围，softmax 曲线平滑、梯度健康。去掉后大维度模型容易训练不稳定甚至不收敛。

> [!question] Q3：Self-Attention 和 Cross-Attention 的本质区别是什么？BEVFormer 里分别用在哪里？
> 提示：看 Q 和 K/V 是否同源。

> [!success]- 参考答案
> Self-Attention 的 Q、K、V 同源（一个序列内部交流）；Cross-Attention 的 Q 和 K/V 异源（两个序列对话）。BEVFormer 里 Temporal Self-Attention 让当前 BEV query 和历史 BEV 特征自己融合（同源）；Spatial Cross-Attention 让 BEV query（Q）从多相机图像特征（K、V）提取信息（异源）。

> [!question] Q4：标准 Attention 是 O(N²) 复杂度，Deformable Attention 怎么降下来的？为什么偏移量要可学习？
> 提示：从"看多少个点"和"投影不准"两个角度。

> [!success]- 参考答案
> 标准 Attention 每个 query 要看全部 N 个位置；Deformable Attention 只在参考点周围采样 K（4-8）个点，复杂度降到 O(N×K)。偏移量可学习是因为投影点不一定是最优采样位置——标定误差、透视畸变、物体运动都会使正确采样点偏离投影点，让网络自己学偏移就是在补偿这些偏差。

> [!question] Q5：为什么必须加位置编码？BEV 里用 3D 坐标做位置编码（PETR）有什么好处？
> 提示：Attention 对顺序/位置是否敏感？

> [!success]- 参考答案
> Attention 本质是集合操作，打乱 token 顺序结果不变，完全没有位置/空间概念，必须显式注入位置信息。BEV 网格自带物理坐标，用 3D 坐标做位置编码（PETR）可以让网络直接学到"图像特征在 3D 空间的哪里"，隐式完成视角变换，省掉显式投影——这是 PETR 能成为 BEV 主流范式之一的核心原因。

---

## 🛠 动手练习

### 练习 1：手算一次完整的 Attention（20 分钟）

给定 2 个 token 的输入 `X = [ [1, 0], [0, 1] ]`（2×2 矩阵，d=2），设：
- `W_Q = W_K = W_V = I`（单位矩阵，简化计算）
- 即 `Q = K = V = X`

手算：
1. `Q·Kᵀ`（2×2 矩阵）
2. 除以 √2 后的值
3. 每行 softmax
4. 用注意力权重加权 V，得到输出

> [!tip] 算完你会发现
> 两个 token 互不相同的输入（一个偏"x 方向"、一个偏"y 方向"），注意力输出后每个 token 变成了两个方向的**加权混合**——这就是"信息交换"的直观体现。

### 练习 2：用 PyTorch 实现一个注意力头（30-60 分钟）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """Q, K, V: [batch, seq_len, d_k]"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, N, N]
    weights = F.softmax(scores, dim=-1)                           # 按行归一化
    return torch.matmul(weights, V)                               # 加权求和

class SelfAttentionHead(nn.Module):
    """单头自注意力：输入 [B, N, d_model]，输出 [B, N, d_model]"""
    def __init__(self, d_model=256, d_k=32):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)

    def forward(self, x):           # x: [B, N, 256]
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        return scaled_dot_product_attention(Q, K, V)

# 验证：输入 4 个 token，看输出形状是否正确
x = torch.randn(2, 4, 256)
out = SelfAttentionHead()(x)
assert out.shape == (2, 4, 32)
print("形状正确:", out.shape)
```

> [!tip] 做完后自问
> ① 把 `scores` 除以 d_k 前打印一下数值范围，再对比除后的——体会 √d_k 的作用。② 改成 8 头（Multi-Head），拼回去维度对不对？③ 用 Cross-Attention 的思路改一版：Q 来自序列 A，K/V 来自序列 B。

### 练习 3：可视化注意力（进阶，可选）

用 `torch` 生成一个小序列，跑上面实现的 Attention，把 `weights`（注意力矩阵）画成热力图（`matplotlib.imshow`）。观察：对角线为什么最亮？如果输入两个相同 token，它们之间会怎样？

---

## ➡️ 下一步学什么

按知识库学习路径，读完本篇你应该接着：

1. **[[3D视觉与投影几何]]** —— BEV 里 Cross-Attention 的"投影采样"依赖相机模型和坐标变换，数学基础在这里。
2. **[[Transformer进阶知识]]** —— 想深入时再看：FlashAttention、RoPE、MoE、GQA 这些工程进阶。
3. **[[BEVFormer详解]]** —— 直接看 Transformer 在 BEV 感知中的完整落地（本篇第四节的展开版）。
4. **[[PETR系列]]** —— 对比另一种思路：用 3D 位置编码替代显式投影。
5. **[[UniAD详解]]** —— 看 Object Query 如何在端到端框架里串联检测→跟踪→规划。

> 💡 自测题 Q4 提到 Deformable Attention 是 BEVFormer 的工程关键——带着这个问题去读 [[BEVFormer详解]]，收获会翻倍。

---

## 相关笔记

- [[计算机视觉基础]] — CNN 基础，理解 Transformer 之前先有"局部 vs 全局"的对比
- [[Transformer进阶知识]] — FlashAttention、RoPE、MoE、Swin、GQA
- [[BEVFormer详解]] — 时空 Transformer BEV 特征变换
- [[PETR系列]] — 3D 位置编码隐式视角变换
- [[UniAD详解]] — 端到端多任务 Query 交互
