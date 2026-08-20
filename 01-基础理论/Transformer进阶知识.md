---
tags: [foundation, transformer, attention, advanced]
created: "2026-07-28"
updated: "2026-08-21"
---

# Transformer 进阶知识

> 一句话导读：基础篇讲"Transformer 是什么、注意力怎么算"，本篇讲**"怎么让它更快、更省、更聪明"**——高效注意力（FlashAttention）、现代位置编码（RoPE/ALiBi）、视觉变体（Swin）、稀疏注意力、MoE 专家混合、GQA 分组查询，最后落到**驾驶模型里到底用哪种**。面试问"你的模型怎么提速？为什么用 Deformable？"就看这篇。

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | FlashAttention 与 Online Softmax、RoPE/ALiBi 位置编码、ViT→Swin→PVT 视觉变体、稀疏注意力家族、MoE 混合专家、GQA 分组查询注意力、驾驶模型中的实战选型 |
| **需要的前置知识** | [[Transformer架构详解]]（⭐ 先读基础篇：Attention/QKV/多头）、[[模型架构演进]]（Transformer 在演进谱系中的位置）|
| **学习顺序建议** | ⭐ 在 [[Transformer架构详解]] 之后读——本篇是"工程/效率"进阶篇，很多内容（Deformable、FlashAttention）到 BEV/端到端章节才用得上，读不懂可以先跳过细节、回头再补 |
| **学完之后你能** | ① 说出 FlashAttention 解决的核心瓶颈（显存带宽）与原理关键词；② 区分 RoPE/ALiBi 各自解决什么问题；③ 解释 Swin 为什么用"窗口+偏移"两段式；④ 看懂 BEVFormer 用 Deformable、PETR 用 3D PE 的选型理由 |
| **预计阅读时间** | 45-60 分钟（分两次读完更轻松：1-3 节一次，4-7 节一次）|

> [!tip] 用一个"图书馆查资料"的比喻理解本篇主线
> 标准 Attention 像在**整座图书馆**里逐本翻书找资料（O(N²)，越多人越慢）；本篇所有技巧都是在想"怎么少翻几本还能找全"：
> - **FlashAttention** = 把常用书搬到**手边的小书桌（SRAM）**上查，不每次跑回大书库（HBM）；
> - **Swin 窗口** = 先在自己**座位周围**查（局部窗口），再站起来换个角度查（偏移窗口），兼顾效率与全局；
> - **MoE** = 找资料时只喊**相关领域的几位专家**来，不把所有管理员都叫来；
> - **GQA** = 几位读者**共用同一份索引卡片**（共享 K,V），省地方（KV cache）。
> 记住这条主线，本篇 7 节就串起来了。

---

## 一、高效注意力机制

> [!note] 大白话：Attention 的瓶颈不是"算不过来"，是"搬数据搬不过来"
> 算 QK^T 的乘法其实很快，真正慢的是**把中间结果（N×N 的矩阵）在显存（HBM）和芯片缓存（SRAM）之间来回搬**。N=4096 时一个中间矩阵就 32MB，搬一次比算一次还贵。**FlashAttention 的绝招：把计算拆成小块（tiling），让每块数据留在 SRAM 里一次算完，不搬出去**（配合 Online Softmax 增量式算归一化 + 反向时重算不缓存）。结果：显存访问从 O(N²) 降到约 O(N√d)——**速度提升不是靠更快的乘法，是靠更少的搬运**。

### 1. FlashAttention — IO-Aware 注意力

**核心问题**：标准 Attention 的瓶颈不是 FLOPS，是显存带宽。

```
标准 Attention 的显存访问:
  S = QK^T     → HBM 写 [N²] (大瓶颈!)
  P = softmax(S) → HBM 写 [N²]
  O = PV        → HBM 读 [N²] + 写 [Nd]

  N=4096, d=64, FP16 下:
    S/P 各需要 4096² × 2 bytes = 32 MB (放不进 SRAM)
    → 每次 softmax 都读 HBM → 带宽瓶颈

Flash Attention:
  核心: Tiling + Online Softmax + Kernel Fusion
  1. 将 Q,K,V 分块 (block size = B, 通常 128)
  2. 在 SRAM 中处理每个 tile 的 softmax (增量式计算)
  3. Backward 时 recompute attention (不存 S,P)
  4. 单 CUDA kernel 完成所有操作
  
  结果: HBM 读写从 O(N²) 降至 O(N²d²/4M) ≈ O(N√d)
        M = SRAM 大小 (A100: 192KB per SM)
```

**关键数据**：

| 版本 | 年份 | 加速比 (训练) | 显存节省 | 支持架构 |
|------|------|-------------|----------|---------|
| FlashAttention v1 | 2022 | 2-4x | 10-20x | Ampere, Hopper |
| FlashAttention v2 | 2023 | 2x over v1 | 同 v1 | H100 优化 |
| FlashAttention v3 | 2024 | 1.5-2x over v2 | 同 v1 | H100 FP8 |

### 2. Online Softmax

```python
# 标准 softmax 需要 3 次 HBM 读写:
m = max(x)           # read x
exp_x = exp(x - m)   # read x, write exp_x
s = sum(exp_x)       # read exp_x
y = exp_x / s        # read exp_x, write y

# Online (增量) Softmax — 一次扫描:
# 利用关系: softmax(x) = softmax(max(x_old, x_new)) * adjustment
m = -inf
s = 0
for x_i in tiles:           # 逐块处理
    m_new = max(m, max(x_i))
    s = s * exp(m - m_new) + sum(exp(x_i - m_new))
    m = m_new
y_i = exp(x_i - m) / s
```

### 3. 各类 Attention 复杂度对比

| 方法 | 时间复杂度 | 空间复杂度 | 是否近似 |
|------|-----------|-----------|----------|
| **Standard** | O(N²d) | O(N²) | 精确 |
| **FlashAttention** | O(N²d) | O(N) | 精确 |
| **Deformable** | O(NKd) | O(NK) | 近似 (K 个采样点) |
| **Linear (Performer)** | O(Nd²) | O(Nd) | 近似 (核函数) |
| **Sparse (BigBird)** | O(N(w+g)d) | O(N(w+g)) | 局部+全局稀疏 |
| **Swin (Window)** | O(M²Nd) | O(M²N) | 局部窗口 |

---

## 二、位置编码深度

> [!note] 大白话：位置编码是给 Transformer 的"座位号"
> Attention 本身**不分先后**（把句子倒过来算出的注意力一样），必须靠位置编码告诉它"谁在第几个位置"。RoPE 和 ALiBi 是 2023-2025 大模型的标配：
> - **RoPE（旋转位置编码）**：把位置信息"旋转"进 Q/K 向量里，最后点积结果**只依赖相对位置（m-n）**——"第 5 个词看第 3 个词"和"第 105 个词看第 103 个词"的关系一样，还能外推到更长序列。缺点：不区分绝对位置（t=0 和 t=1 的"角色"分不清）。
> - **ALiBi（线性偏置）**：更简单粗暴——直接在注意力分数上加一个"距离越远、惩罚越大"的线性项，**零参数**、外推性好。
> - 记住：**RoPE 是"把位置编码进向量"，ALiBi 是"在打分时按距离扣分"。**

### RoPE (Rotary Position Embedding)

```
核心公式:
  f_q(x_m, m) = R_m^Θ · W_q · x_m
  f_k(x_n, n) = R_n^Θ · W_k · x_n

其中 R_θ,m = 
  [cos mθ_1, -sin mθ_1, 0, 0, ...]
  [sin mθ_1,  cos mθ_1, 0, 0, ...]
  [0, 0, cos mθ_2, -sin mθ_2, ...]
  [0, 0, sin mθ_2,  cos mθ_2, ...]
  
  频率: θ_i = 10000^(-2i/d)

点积结果:
  q_m^T · k_n = (R_m^Θ · q)^T · (R_n^Θ · k)
              = q^T · R_{n-m}^Θ · k
              = f(q, k, m-n)  ← 只依赖于相对位置!

RoPE 特性:
  ✅ 相对位置天然编码
  ✅ 可外推到更长序列 (通过插值 θ)
  ✅ 训练/推理一致性
  ❌ 不包含绝对位置信息 (不能区分 token 0 vs token 1)
```

### ALiBi (Attention with Linear Biases)

```python
# 直接在 attention score 上加线性衰减偏置
# 替代位置编码的最简单方法

def attention_with_alibi(Q, K, V, slopes):
    """
    slopes: 每个 head 的斜率，如 2^(-8/n_heads * head_idx)
    """
    scores = Q @ K.T / sqrt(d_k)
    
    # 计算距离矩阵 → 线性偏置
    distances = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    # 负距离 (过去) → 正偏置 (更多关注)
    alibi_bias = -slopes * distances.abs()  # 或更精细的处理
    
    scores = scores + alibi_bias
    return softmax(scores) @ V

优势: 零参数, 快速外推, 训练时短序列 → 推理时长序列自然适配
```

---

## 三、视觉 Transformer 变体

> [!note] 大白话：图像太大，全局 Attention 太贵 → 三个应对思路
> 图像做成 patch 后数量巨大，全局 O(N²) 直接爆炸。三招：**① Swin：局部窗口**——只在 7×7 小窗口内做注意力（便宜），再用"窗口偏移"让相邻窗口信息流通（看全局）；**② PVT：空间缩减**——把 K,V 用卷积压缩再算；**③ Focal：远近分工**——近处细看、远处粗看，像人眼。**Swin 的"偏移窗口"是核心考点**：普通窗口互相不通信（A 看不到 B），偏移后相邻窗口的边界信息能跨窗口流动，还不增加计算量。

### ViT 到 Swin 的演进

```
ViT (ICLR 2021):
  图像 → 16×16 patches → Linear Projection → Transformer Encoder → CLS token → 分类
  问题: 全局 Attention 为 O(N²), 缺乏层次化特征, 需要 JFT-300M 预训练

DeiT (ICML 2021):
  学生模型从 CNN 教师蒸馏 → 可在 ImageNet-1K 训练 (不需大数据)
  引入 distillation token + hard label distillation

Swin Transformer (ICCV 2021, 马尔奖):
  层次化金字塔结构:
    Stage1: 4×4 patch, window=7 → [H/4, W/4, C]
    Stage2: 8×8 patch, window=7 → [H/8, W/8, 2C]  
    Stage3: 16×16 patch, window=7 → [H/16, W/16, 4C]
    Stage4: 32×32 patch, window=7 → [H/32, W/32, 8C]

  W-MSA (Window MSA): 在 7×7 窗口内 Attention → O(49² × N)
  SW-MSA (Shifted Window): 窗口偏移 (3,3) → 跨窗口通信

PVT (Pyramid Vision Transformer, ICCV 2021):
  空间缩减 Attention (SRA): 用卷积压缩 K,V 的空间分辨率
  → O((H/r × W/r)²) 复杂度
  
Focal Transformer (NeurIPS 2021):
  细粒度关注近处 + 粗粒度关注远处
  → 类似人的视觉注意机制
```

### Swin 的 Shifted Window 详解

```
W-MSA (Window Multi-head Self-Attention):
  ┌─────┬─────┬─────┐
  │  A  │  B  │  C  │  ← 每个窗口内独立 Self-Attention
  ├─────┼─────┼─────┤
  │  D  │  E  │  F  │    问题: 窗口间无信息交换!
  ├─────┼─────┼─────┤
  │  G  │  H  │  I  │
  └─────┴─────┴─────┘

SW-MSA (Shifted Window MSA):
  ┌──┬───┬───┬───┬──┐
  │A1│   │   │   │C1│  ← 窗口偏移 (⌊M/2⌋, ⌊M/2⌋)
  ├──┼───┼───┼───┼──┤    产生 9 个新窗口 (1个4倍大小)
  │  │   │   │   │  │
  ├──┼───┼───┼───┼──┤  Cyclic Shift + Mask:
  │  │   │   │   │  │    将 9 个窗口 → 拼成 4 个大方块
  ├──┼───┼───┼───┼──┤    用 attention mask 防止跨窗口注意
  │  │   │   │   │  │
  ├──┼───┼───┼───┼──┤  效率不变 (仍是窗口 Attention)
  │G1│   │   │   │I1│
  └──┴───┴───┴───┴──┘
```

---

## 四、稀疏/高效注意力

### Sparse Attention 家族

```python
# 1. Local + Global (Longformer, BigBird)
# 每个 token 关注: 周围 w 个局部 token + g 个全局 token
# 复杂度: O(N(w+g))

# 2. Block Sparse
# 只计算预定义的 block 间 attention
# 对结构化数据 (如图像 patch) 特别有效

# 3. Reformer (LSH Attention)
# 用 Locality Sensitive Hashing 找到最相似的 Q-K 对
# 只计算相似度高的 token 对

# 4. Linformer
# 将 K,V 的序列维度 N 投影到固定维度 k
# 复杂度: O(N × k × d), k 是常数 (如 256)
```

### 在驾驶感知中的应用选择

| 场景 | 推荐 Attention | 原因 |
|------|---------------|------|
| BEV 空间 (128×128) | Deformable | 只需关注投影区域附近 |
| 时序融合 (3-5帧) | Full Self-Attn | 序列短，O(25) 可接受 |
| 图像 Backbone | Swin Window | 层次化 + 高效 |
| 大模型推理 | FlashAttention | 训练/推理加速 |
| Object Query 交互 | Cross-Attention | Q=query, KV=特征 |

---

## 五、MoE (Mixture of Experts)

> [!note] 大白话：MoE = "专家门诊，按需叫号"
> 大模型 90% 的参数平时都在"摸鱼"（每个 token 不需要全部能力）。MoE 加一个**路由器（Router）**，每个 token 只叫 **Top-K 个专家**来干活：Mixtral 8×7B 总参数 56B，但每个 token 只激活 2 个专家 = 14B 参数量——**参数多 8 倍、算力只多一点点，能力却接近大模型**。三个工程痛点：① 负载均衡（别让一个专家累死，别的闲着）；② 专家坍缩（全路由到同一个专家 = 白做）；③ 跨卡通信（专家在不同 GPU 上，all-to-all 通信贵）。智驾里可做"城市/高速/泊车不同场景各配专家"。

### 关键问题

```
核心思想: 不是所有参数在处理每个 token 时都有用
          → 动态选择部分专家 (Expert)

架构:
  Token → Router (gate) → Top-K Experts → Weighted Sum

  Router: g(x) = softmax(x · W_r)  → 选 Top-K
  Expert: FFN (通常: Linear → GeLU → Linear)
  Output: y = Σ g_i(x) × Expert_i(x), i ∈ Top-K

配置示例 (Mixtral 8×7B):
  Experts: 8 个
  Top-K: 2 (每个 token 激活 2 个 Expert)
  总参数: 8 × 7B = 56B (但每个 token 只用 2 × 7B = 14B)

关键问题:
  1. Load Balancing: 防止某些 expert 过载
     → Load Balancing Loss = K² Σ f_i · P_i
  2. Expert Collapse: 所有 token 都被路由到同一个 expert
     → 需要 auxiliary loss 鼓励均匀分布
  3. Communication: Expert 散布在不同 GPU → all-to-all 通信

在自动驾驶中的潜力:
  - 多场景切换 (城市/高速/泊车) → 不同 expert 激活
  - 不同天气/光照 → MoE 自适应路由
  - 大规模多任务模型 → 不同 task 的 expert
```

---

## 六、GQA (Grouped Query Attention)

> [!note] 大白话：GQA = "几个人共用一份笔记"
> 推理时要缓存每个 token 的 K,V（**KV cache**），序列越长缓存越大，直接决定推理吞吐。**MHA**（多头）每个头各存一份 → 缓存最大；**MQA**（多查询）所有头共用一份 → 缓存最小但质量掉；**GQA**（分组查询）**把 h 个头分成 g 组，每组内共享 K,V**——质量接近 MHA、效率接近 MQA，是 LLaMA 2/3 系的标准选择（70B 用 h=64, g=8，KV cache 压缩 8 倍）。面试口诀：**质量 MHA > GQA > MQA，效率反过来。**

```
Multi-Head (MHA):
  Q: h 个头, K: h 个头, V: h 个头
  每对独立的 Q,K,V (参数量大, KV cache 大)

Multi-Query (MQA):
  Q: h 个头, K: 1 个共享, V: 1 个共享
  所有 head 共享 K,V → KV cache 缩小 h 倍, 但质量下降

Grouped Query (GQA):
  Q: h 个头
  K,V: g 组 (1 < g < h), 每个 group 内有 h/g 个 head 共享 K,V
  例: h=32, g=8 → 每 4 个 Q head 共享一组 K,V

Trade-off:
  MHA > GQA > MQA  (质量)
  MQA > GQA > MHA  (效率, 尤其在长序列推理)

LLaMA 2 70B: h=64, g=8 (8x KV cache 压缩)
```

---

## 七、在驾驶模型中的实战应用

### 各模型的选择

| 模型 | Attention 类型 | Backbone | 关键优化 |
|------|---------------|----------|----------|
| **BEVFormer** | Deformable Cross-Attn + Temporal Self-Attn | ResNet/Swin | Deformable O(NK) |
| **BEVDet** | LSS (无 Attention) | ResNet | 纯 CNN → 高效 |
| **PETR** | Cross-Attn + 3D PE | ResNet/ViT | 3D PE 质量关键 |
| **UniAD** | Multi-Query Cross-Attn | VoVNet/ViT | Query 间 Self-Attn |
| **VAD** | Vectorized Attention | ResNet | 稀疏表示 |

### 训练加速建议

```python
# BEV 模型训练时的 Attention 优化
from flash_attn import flash_attn_func

# 替换标准 Attention
def efficient_attention(Q, K, V):
    # 短序列 (<512): 标准 Attention (开销可接受)
    # 长序列 (>512): Flash Attention
    if Q.shape[1] > 512:
        return flash_attn_func(Q, K, V, causal=False)
    else:
        return F.scaled_dot_product_attention(Q, K, V)
```

> [!note] 大白话：驾驶模型选 Attention 的口诀
> 表格记不住？记三条：**① BEV 空间用 Deformable**（128×128 网格太大，全局注意没必要，只看投影区域附近——BEVFormer 的核心）；**② 短时序（3-5 帧）用全量 Self-Attn**（序列太短，O(25) 便宜得很）；**③ 图像 Backbone 用 Swin 窗口**（层次化 + 高效）。大模型推理统一 FlashAttention。PETR 的"3D PE"是另一个流派：不采样、直接用 3D 位置编码注入 Query——记住 **BEVFormer 采样 vs PETR 编码** 这对概念，面试常考。

---

## ✅ 自测一下（先想答案，再点开）

> [!question] Q1：标准 Attention 的真正瓶颈是什么？FlashAttention 怎么解决？

> [!success]- 参考答案
> 瓶颈不是算力（FLOPS）而是**显存带宽**——中间结果 S/P（N×N）要在 HBM 和 SRAM 之间来回搬（N=4096 时单个矩阵 32MB，放不进 SRAM）。FlashAttention 用 **Tiling（分块）+ Online Softmax（增量式归一化）+ Kernel Fusion（单 kernel）**，让数据留在 SRAM 里一次算完；反向时**重算不缓存** S/P。结果：HBM 访问从 O(N²) 降到约 O(N√d)，训练加速 2-4x、显存节省 10-20x，且**仍是精确计算**（不是近似）。

> [!question] Q2：RoPE 为什么"天然编码相对位置"？它有什么局限？

> [!success]- 参考答案
> RoPE 把位置信息以旋转矩阵 R_m 的形式乘进 Q/K，点积后旋转角抵消，结果**只依赖位置差 (m-n)**——相对位置天然编码（q_m·k_n = q·R_{n-m}·k）。局限：**不含绝对位置信息**（区分不了 t=0 和 t=1 的角色差异）；可外推靠对频率 θ 插值。ALiBi 是另一条路：不加编码、直接在 score 上加"距离越远扣分越多"的线性偏置，零参数。

> [!question] Q3：Swin 为什么需要"窗口偏移"（SW-MSA）？

> [!success]- 参考答案
> 纯窗口注意力（W-MSA）每个 7×7 窗口内独立算，**窗口之间无信息交换**（A 窗口看不到 B 窗口）。偏移 (⌊M/2⌋, ⌊M/2⌋) 后窗口边界错位，原边界处的内容进入新窗口中心，实现**跨窗口通信**；配合 Cyclic Shift + Mask 保证计算效率不变（仍是窗口注意力）。窗口注意力把复杂度从 O(N²) 降到 O(M²N)，且天然层次化（金字塔结构），所以适合做图像 Backbone。

> [!question] Q4：MHA / MQA / GQA 的区别？为什么 GQA 是大模型标配？

> [!success]- 参考答案
> 区别在 K,V 的共享方式：MHA 每个头独立（质量最好、KV cache 最大）；MQA 所有头共享一份（KV cache 缩 h 倍、质量下降）；GQA 把 h 头分 g 组、组内共享（h=32,g=8 → 每 4 个 Q 头共享一组 K,V），质量接近 MHA、效率接近 MQA。LLaMA 2 70B 用 h=64,g=8 压缩 8 倍 KV cache——**推理时 KV cache 大小直接决定吞吐**，所以 GQA 是长序列推理标配。

---

## 🛠 动手练习

1. **比一比**：把 3. 各类 Attention 复杂度对比表抄一遍，对每行标注"精确 or 近似"——体会 FlashAttention 是"唯一又快又精确"的。
2. **读代码**：找 [[模型部署与延迟优化]] 或 [[AI-Infra详解]] 里 FlashAttention 的调用方式，理解 `flash_attn_func(Q, K, V, causal=...)` 的参数。
3. **对上号**：打开 [[BEVFormer详解]]，确认它用的是 Deformable Cross-Attn + Temporal Self-Attn；打开 [[PETR系列]] 对比 3D PE 方案——两篇笔记各写一句"选型理由"。
4. **口算 KV cache**：MHA 下序列 4096、h=32、d=128、FP16，KV cache 多大？GQA(g=8) 后多大？（答案：MHA = 2×4096×32×128×2B = 64MB；GQA = 64/8 = 8MB——感受为什么推理要 GQA。）

---

## ➡️ 下一步学什么

本篇是效率/工程进阶，按学习路径：

1. **[[3D视觉与投影几何]]** —— 基础理论最后一站：把 Transformer 能力用到 3D（BEV 变换、投影）前的数学准备。
2. **[[BEV感知全景]]** —— 实战第一站：看 Transformer（Deformable/Cross-Attn）如何在 BEV 感知里大显身手。
3. **[[BEVFormer详解]]** —— 本篇"Deformable + Temporal"选型的完整落地案例。
4. **[[模型部署与延迟优化]]** —— 本篇的 FlashAttention/GQA/MoE 都是部署加速的工具箱，工程面试直接复用。
5. **[[模型架构演进]]** —— 没读先读：本篇的"进阶"建立在架构演进脉络之上。

> 💡 本篇很多知识点（FlashAttention、MoE、GQA）也是大模型/推理优化面试题，读 [[AI-Infra详解]] 和 [[AI-Infra前沿]] 时你会再次遇到它们——一次学会，多处复用。

---

## 相关笔记

- [[Transformer架构详解]] — 基础篇（本篇前置，QKV/多头注意力）
- [[模型架构演进]] — Transformer 在架构演进谱系中的位置
- [[BEVFormer详解]] — Deformable Cross-Attn 落地案例
- [[PETR系列]] — 3D PE 流派对照
- [[模型部署与延迟优化]] — FlashAttention/量化等部署加速
- [[AI-Infra详解]] — MoE/GQA 的分布式训练视角
- [[智驾算法面试题库]] — 面试实战
