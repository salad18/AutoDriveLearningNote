---
tags: [foundation, transformer, attention]
created: "2026-07-21"
---

# Transformer 架构详解

> Attention Is All You Need. Transformer 彻底改变了 CV 和自动驾驶感知。

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

---

## 二、核心机制

### 1. Self-Attention（自注意力）

```
输入: X ∈ R^{N×d}
Q = X·W_Q    (Query: 我要查什么)
K = X·W_K    (Key: 我能提供什么)
V = X·W_V    (Value: 我实际有什么)

Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```

**直观理解**：每个 token 都去问所有其他 token "你和我有多大关系？"，然后加权聚合信息。

**`√d_k` 缩放**：防止点积过大导致 softmax 梯度消失。

### 2. Multi-Head Attention（多头注意力）

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
其中 head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

- 不同头关注不同的关系：位置、语义、几何等
- 典型配置：`d_model=256, h=8, d_k=32`

### 3. Cross-Attention（交叉注意力）

```
Q 来自一个序列（如 BEV Query）
K, V 来自另一个序列（如图像特征）
```

**BEVFormer 中的用法**：
- BEV Query 作为 Q
- 图像特征作为 K, V
- 通过相机内外参，将 BEV Query 的 3D 位置投影到 2D 图像上采样

### 4. Deformable Attention（可变形注意力）

标准 Attention 是 O(N²) 复杂度，Deformable Attention 只采样 K 个关键点：

```
DeformAttn(Q, p, x) = Σ W·x(p + Δp)  # 在参考点附近采样 K 个点
```

- **复杂度**：O(N × K)，K 通常 4-8
- **在 BEV 中的应用**：BEVFormer 的 Spatial Cross-Attention

### 5. Position Encoding（位置编码）

| 类型 | 方法 | 使用场景 |
|------|------|----------|
| **正弦位置编码** | sin/cos 函数 | 原始 Transformer |
| **可学习位置编码** | 参数化向量 | ViT, BEVFormer |
| **3D 位置编码** | 3D 坐标 → MLP → PE | PETR |
| **相机参数编码** | 内外参 → 编码 | BEV 特征变换 |

---

## 三、经典变体

### ViT (Vision Transformer)

```
图像 → 切分成 Patches → Patch Embedding → Transformer Encoder → 分类
```

- **核心思想**：把图片当作 token 序列处理
- **局限性**：计算量大，需要大量数据预训练

### DETR (DEtection TRansformer)

```
图像 → CNN Backbone → Transformer Encoder → Object Queries → Transformer Decoder → 预测框
```

- **核心创新**：二分图匹配（Hungarian 匹配）替代 NMS
- **Object Query**：可学习的检测询问向量
- **在自动驾驶中**：DETR3D 将 Object Query 扩展到 3D 空间

### Swin Transformer

- 层次化结构，适合作为视觉 Backbone
- 窗口 Attention（局部）+ 移位窗口（跨窗口通信）

---

## 四、BEV 感知中的 Transformer

### BEVFormer 的时空 Attention

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

### 关键设计

1. **BEV Query** = 可学习的位置锚点，分布在鸟瞰网格上
2. **参考点投影**：BEV Grid (x,y) → 3D 参考点 → 相机投影 → 2D 采样位置
3. **时序对齐**：历史 BEV 特征根据 ego 运动对齐到当前坐标系

---

## 五、关键概念速记

| 概念 | 含义 |
|------|------|
| **Q/K/V** | Query/Key/Value 三元组 |
| **Cross-Attention** | Q 和 K/V 来自不同源 |
| **Self-Attention** | Q、K、V 来自相同源 |
| **Deformable Attn** | 稀疏采样，降低计算量 |
| **FFN** | Feed-Forward Network，位置独立 MLP |
| **LayerNorm** | 层归一化，稳定训练 |
| **Object Query** | DETR 系的可学习检测锚点 |

---

## 📖 推荐资料

- 原始论文：Attention Is All You Need (2017)
- The Illustrated Transformer (Jay Alammar)
- DETR 论文 + 代码
- BEVFormer 论文 Section 3（Spatial/Temporal Attention 详解）

---

## 相关笔记

- [[计算机视觉基础]]
- [[BEVFormer详解]]
- [[PETR系列]]
- [[UniAD详解]]
