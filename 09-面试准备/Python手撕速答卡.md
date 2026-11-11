---
tags: [interview, 手撕, python, 速答]
created: "2026-11-11"
updated: "2026-11-11"
---

# Python 手撕速答卡（面试前 10 分钟）

> 不是教程：**只给能默写的骨架 + 边界坑 + 追问**。面试前 10 分钟按顺序过一遍，白板能写出来就够。
> 需要理解原理 → [[手写算法-Python版]]。

---

## 〇、10 分钟怎么过

```
第 1 分钟：Attention（最常考）
第 2 分钟：NMS
第 3 分钟：IoU
第 4 分钟：Softmax（数值稳定）
第 5 分钟：BN/LN/GN 三行区别
第 6 分钟：Focal Loss
第 7-8 分钟：Top-k / 匈牙利匹配（数据闭环常用）
第 9-10 分钟：把下面"边界坑"表扫一遍
```

---

## 一、Self-Attention（含 Multi-Head）

```python
import numpy as np

def attention(Q, K, V, mask=None):
    # Q:[B,H,Lq,d]  K,V:[B,H,Lk,d]
    d = Q.shape[-1]
    S = Q @ K.transpose(0, 1, -1, -2) / np.sqrt(d)      # [B,H,Lq,Lk]
    if mask is not None:
        S = np.where(mask, S, -1e9)                      # 加性 mask
    S = S - S.max(-1, keepdims=True)                     # 数值稳定
    A = np.exp(S)
    A = A / A.sum(-1, keepdims=True)
    return A @ V

def mha(X, Wq, Wk, Wv, Wo, h):
    B, L, D = X.shape
    q, k, v = X @ Wq, X @ Wk, X @ Wv
    split = lambda t: t.reshape(B, L, h, -1).transpose(0, 2, 1, 3)   # [B,h,L,dh]
    out = attention(split(q), split(k), split(v))                    # [B,h,L,dh]
    out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
    return out @ Wo
```

- **一句话原理**：用 Q·K 算相关性，softmax 成权重，加权求和 V。
- **坑**：`mask` 里 `True` 表示**保留**；`-1e9` 不用 `-inf`（`-inf` 减 max 会 NaN）；`/sqrt(d)` 不能省。
- **追问**：为什么除以 √d？→ 点积方差随 d 增大，softmax 会饱和、梯度消失。MHA 怎么切？→ reshape 成 h 段再 transpose，算完 concat。

---

## 二、NMS（标准 + Soft-NMS 一句话）

```python
def iou_xyxy(a, b):                       # a:[4], b:[N,4]
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)

def nms(boxes, scores, thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= thr]     # 保留 IoU 低的
    return np.array(keep)
```

- **坑**：`+1e-9` 防除零；`boxes` 要 float；3D NMS 的 IoU 用 BEV IoU × 高度重叠。
- **追问**：复杂度？→ 最坏 O(N²)，工程上用分数阈值预筛/GPU。Soft-NMS 改什么？→ 不直接删，用 `score *= (1-IoU)` 或高斯衰减。

---

## 三、IoU（2D / 旋转框 / 3D 一句话）

- 2D 轴对齐：上面 `iou_xyxy`。
- **旋转框**：多边形相交（Sutherland–Hodgman）算交面积；工程上可先转 mask 近似。
- **3D**：`IoU_3d = IoU_bev × (h_overlap / (h1 + h2 - h_overlap))`。
- **追问**：IoU 的缺点？→ 不相交时为 0、无梯度 → 用 GIoU/DIoU/CIoU。

---

## 四、Softmax（数值稳定版）

```python
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
```

- **坑**：先减 max 再 exp；分母加 `1e-9` 更稳。
- **追问**：和 log-softmax 关系？→ `log_softmax = x - logsumexp(x)`，避免先算概率再取 log 的精度损失。

---

## 五、BN / LN / GN（三行区别）

```python
def bn(x, g, b, eps=1e-5):          # x:[N,C,H,W]  按 (N,H,W) 归一化
    mu = x.mean((0, 2, 3), keepdims=True)
    var = x.var((0, 2, 3), keepdims=True)
    return g.reshape(1, -1, 1, 1) * (x - mu) / np.sqrt(var + eps) + b.reshape(1, -1, 1, 1)

def ln(x, g, b, eps=1e-5):          # x:[...,D]    按最后一维归一化
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def gn(x, g, b, G=32, eps=1e-5):    # x:[N,C,H,W]  按 (C/G,H,W) 分组归一化
    N, C, H, W = x.shape
    x = x.reshape(N, G, C // G, H, W)
    mu = x.mean((2, 3, 4), keepdims=True)
    var = x.var((2, 3, 4), keepdims=True)
    x = (x - mu) / np.sqrt(var + eps)
    return x.reshape(N, C, H, W) * g.reshape(1, C, 1, 1) + b.reshape(1, C, 1, 1)
```

- **一句话**：**BN 跨样本、LN 跨特征、GN 跨分组**（小 batch 用 GN/LN）。
- **坑**：BN 训练用 batch 统计、推理用 running stats；LN 不需要 running stats；**GN 的 G 必须整除 C**。
- **追问**：Transformer 为什么用 LN？→ 与 batch 无关、序列长度可变。

---

## 六、Focal Loss

```python
def focal_loss(p, y, alpha=0.25, gamma=2.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pt = np.where(y == 1, p, 1 - p)
    at = np.where(y == 1, alpha, 1 - alpha)
    return (-at * (1 - pt) ** gamma * np.log(pt)).mean()
```

- **一句话**：`-(1-pt)^γ` 压低简单样本，`α` 平衡正负类。
- **坑**：`p` 要 clip；多分类用 softmax 版（对每个类算）。
- **追问**：γ=0 退化成什么？→ 带 α 的交叉熵。

---

## 七、Top-k / 匈牙利匹配（数据闭环最常用）

```python
def topk(x, k):
    k = min(k, x.size)
    if k == x.size:
        return np.argsort(-x)
    idx = np.argpartition(-x, k)[:k]      # O(n)，不要求有序
    return idx[np.argsort(-x[idx])]       # 再对 k 个排序

from scipy.optimize import linear_sum_assignment
def match(cost):                          # cost:[N,M]
    r, c = linear_sum_assignment(cost)    # 最小代价匹配
    return r, c
```

- **坑**：`argpartition` 取的是第 k 小，取 top-k 要 `-x`；匹配要求 cost 是"代价"（越小越优）。
- **追问**：DETR 为什么用匈牙利？→ 预测与 GT 是**无序集合**，要一对一匹配；代价通常 = 分类 + L1 + GIoU。

---

## 八、RoPE（会写就加分）

```python
def rope(x, base=10000):                  # x:[..., L, D]，D 为偶数
    D = x.shape[-1]
    pos = np.arange(x.shape[-2])[:, None]
    theta = base ** (-np.arange(0, D, 2) / D)
    ang = pos * theta                     # [L, D/2]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack([x1 * cos - x2 * sin,
                     x1 * sin + x2 * cos], axis=-1).reshape(x.shape)
```

- **一句话**：把 q/k 按位置旋转，点积只依赖相对位置差。
- **坑**：偶数维配对；cos/sin 的 shape 要能广播到 `[...]`。

---

## 九、边界坑表（写完自查）

| 坑 | 怎么防 |
|----|--------|
| 空输入 / 单元素 | 先判 `len==0`，NMS 里 `order.size==1` 提前 break |
| 除零 | 分母 `+1e-9` |
| mask 用 `-inf` | 用 `-1e9`，否则 `-inf - max` 出 NaN |
| dtype | 统一 float32，索引用 int |
| 广播维度 | 归一化时 `keepdims=True` |
| in-place | 别改输入（面试官可能复用） |

---

## 十、面试官最爱的 12 个追问（一句话答案）

| # | 追问 | 一句话 |
|:--:|------|--------|
| 1 | 为什么 `/sqrt(d)`？ | 控制点积方差，防 softmax 饱和 |
| 2 | mask 怎么加？ | 加性 `-1e9`（或 `-inf` 的自定义实现） |
| 3 | MHA 怎么切头？ | reshape 成 h 段 + transpose，算完 concat |
| 4 | NMS 复杂度？ | 最坏 O(N²)，工程上预筛/并行 |
| 5 | Soft-NMS 改什么？ | 用 IoU 衰减分数，不直接删框 |
| 6 | IoU 的缺点？ | 无重叠时无梯度 → GIoU/DIoU |
| 7 | BN 推理用什么统计？ | running mean/var |
| 8 | LN 为什么适合 Transformer？ | 与 batch 无关、变长友好 |
| 9 | Focal 的 γ 作用？ | 降低简单样本权重 |
| 10 | RoPE 为什么是相对的？ | q·k 只依赖 m−n |
| 11 | 匈牙利匹配的代价？ | 分类 + L1 + GIoU（DETR） |
| 12 | Top-k 怎么 O(n)？ | `argpartition` 再对 k 个排序 |

---

## 相关笔记

- [[手写算法-Python版]] — 6 件套完整实现 + 追问（理解版）
- [[LeetCode刷题规划]] — 算法题清单
- [[面试前72小时清单]] — 上场清单
- [[项目深挖追问链]] — 项目追问
- [[面试速答卡-在线建图-多模态-扩散]] — 前沿速答
