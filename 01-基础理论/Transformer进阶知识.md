---
tags: [foundation, transformer, attention, advanced]
created: "2026-07-28"
---

# Transformer 进阶知识

> 在 [[Transformer架构详解|基础篇]] 之上，深入效率优化、现代变体与应用细节。

---

## 一、高效注意力机制

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

---

> 📚 回到基础: [[Transformer架构详解]]
> 📚 面试实战: [[常见面试题-感知算法]]
