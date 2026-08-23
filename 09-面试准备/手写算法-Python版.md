---
title: "手写算法-Python版"
tags: [interview, coding, python, 手撕]
created: "2026-08-21"
updated: "2026-08-21"
---

# 手写算法 — Python 版（面试手撕 6 件套）

> 一句话导读：感知算法岗的"算法题"不是 LeetCode，而是**模型核心算法的推导 + 白板实现**。本篇把 [[面试准备策略]] 六节列出的 6 类必考算法，用 **numpy 一行行写出来**（不是调库）：Self-Attention / NMS / IoU / Focal Loss / BatchNorm / RoPE——每类都带"为什么考 + 大白话 + 完整代码 + 面试追问"。

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | 6 类必考算法的 Python 手撕（numpy 实现，附逐行注释与面试追问）|
| **需要的前置知识** | [[模型架构演进]]（Attention）、[[Transformer架构详解]]、[[模型基础知识补充]]（Loss/归一化）、[[Transformer进阶知识]]（RoPE）|
| **学习顺序建议** | ⭐ 面试前 2 周每天 1 类；和 [[查漏补缺与求职行动清单]] 🔥 第 1 项配套 |
| **学完之后你能** | ① 不看参考写出 6 类算法；② 说清每个算法的时间/空间复杂度；③ 回答"为什么这样写"的追问 |
| **预计阅读时间** | 每类 20-30 分钟，共 2-3 小时（建议分 2 天）|

> [!tip] 手撕的正确姿势
> **先手写思路（伪代码）→ 再写 numpy → 最后口述复杂度**。面试官看的是：① 能不能从原理推导出实现；② 边界条件（除零、空输入）是否考虑；③ 复杂度是否说清。**别背代码，背"推导链"**——每个算法都有一条"公式 → 伪代码 → 实现"的链，本篇按这条链组织。

---

## 一、Self-Attention 前向（含 Multi-Head）⭐⭐⭐⭐⭐

**为什么考**：Transformer 的基石，90% 面试必出；常要求写出 scaled dot-product attention 并手推复杂度。

**大白话**：Q、K、V 三张表。`Q @ K^T` 算"每对 token 的相似度"（得分），除以 √d 防止得分过大把 softmax 推成 one-hot，softmax 归一化成权重，最后加权 V。**"每个 token 问其他 token：你跟我像不像？像就多参考你的信息。"**

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    返回: (batch, seq_len, d_k)
    复杂度: 时间 O(n²·d), 空间 O(n²)（存 attention matrix）
    """
    d_k = Q.shape[-1]
    # 1. 得分矩阵: (b, n, d) @ (b, d, n) -> (b, n, n)
    scores = Q @ K.transpose(0, 2, 1)          # (b, n, n)
    # 2. 缩放: 防止 d 大时 softmax 梯度消失（得分方差随 d 增大）
    scores = scores / np.sqrt(d_k)
    if mask is not None:                        # 因果 mask: 上三角屏蔽
        scores = np.where(mask, scores, -1e9)   # 用 -inf 会被 softmax 变 NaN，用大负数
    # 3. softmax 沿最后一维（对每个 query 的所有 key 归一化）
    exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))  # 减 max 防溢出
    attn = exp_s / exp_s.sum(axis=-1, keepdims=True)             # (b, n, n)
    # 4. 加权求和
    out = attn @ V                             # (b, n, d)
    return out, attn

# ---- Multi-Head: 先 split 再算再 concat ----
def multi_head_attention(Q, K, V, num_heads=8, Wq=None, Wk=None, Wv=None, Wo=None):
    """简化版：假设输入已是投影后的 (b, n, d_model)"""
    b, n, d_model = Q.shape
    d_head = d_model // num_heads
    # 1. reshape: (b, n, d_model) -> (b, n, h, d_head) -> (b, h, n, d_head)
    Qh = Q.reshape(b, n, num_heads, d_head).transpose(0, 2, 1, 3)
    Kh = K.reshape(b, n, num_heads, d_head).transpose(0, 2, 1, 3)
    Vh = V.reshape(b, n, num_heads, d_head).transpose(0, 2, 1, 3)
    # 2. 每个 head 独立算 attention
    outs = []
    for h in range(num_heads):
        o, _ = scaled_dot_product_attention(Qh[:, h], Kh[:, h], Vh[:, h])
        outs.append(o)                          # (b, n, d_head)
    # 3. concat: (b, n, d_head) x h -> (b, n, d_model)
    return np.concatenate(outs, axis=-1)
```

**面试追问**：
- Q：为什么除以 √d_k？→ 得分是 d 个独立项求和，方差 ~d，除 √d 让 softmax 输入方差回到 1，避免饱和。
- Q：mask 为什么用 -1e9 不用 -inf？→ softmax 里 exp(-inf)=0 会得 NaN（0/0），实践中用大负数。
- Q：空间复杂度为什么是 O(n²)？→ attention 矩阵 (n,n) 要存在内存里（FlashAttention 就是为省它）。

---

## 二、NMS（2D / 3D / Soft-NMS）⭐⭐⭐⭐⭐

**为什么考**：检测后处理标配，**每家都会问**；常要求手写 + 说 Soft-NMS 改进。

**大白话**：一堆框里有很多是"同一个物体的重复检测"。NMS = 按分数从高到低，**留下最高的，删掉所有跟它 IoU 太大的**，重复直到没有框。

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: (N, 4) xyxy 格式 [x1, y1, x2, y2]
    scores: (N,)
    返回: 保留的框的索引列表
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)          # 每个框面积
    order = scores.argsort()[::-1]          # 分数从高到低的索引
    keep = []
    while order.size > 0:
        i = order[0]                        # 当前最高分框
        keep.append(i)
        # 当前框 与 剩余所有框 的交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)      # 无交集时为 0
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # 并集 = A+B-交
        inds = np.where(iou <= iou_threshold)[0]  # 保留 IoU 小的
        order = order[inds + 1]             # +1 因为 order[1:] 偏移了一位
    return keep

def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001):
    """Soft-NMS: 不直接删，而是按 IoU 衰减分数（线性或高斯）"""
    # 高斯版: score_new = score * exp(-iou² / sigma)
    # 优点: 两个高 IoU 框可能是两个真目标（密集场景），直接删会漏检
    # 实现要点: 每次取当前最高分框，对剩余框分数做衰减，低于阈值才移除
    pass  # 思路同上，区别是删改衰减；面试口述思路即可，很少要求完整写
```

**3D NMS（BEV 旋转框）**：框带朝向角 `(x, y, z, w, l, h, theta)`。IoU 计算更麻烦（旋转矩形的交面积），**面试通常只考思路**：把旋转框转成凸多边形 → 求多边形交（Sutherland–Hodgman 裁剪）→ 算面积。

**面试追问**：
- Q：NMS 的复杂度？→ O(N²)（每轮和剩余所有框算 IoU），可用 N 个框排序后提前剪枝优化。
- Q：Soft-NMS 解决了什么？→ 密集场景两个真目标重叠时，NMS 会误删低分真框；Soft-NMS 按 IoU 降分而非删框。
- Q：和匈牙利匹配的区别？→ NMS 是"去重"（同类内），匈牙利是"关联"（跨帧/跨模态配对），见 [[智驾算法面试题库]] Q21。

---

## 三、IoU 计算（2D / 旋转框 / 3D）⭐⭐⭐⭐

**为什么考**：NMS、匹配、Loss 都要用；手写 2D 后常追问旋转框。

**大白话**：两个框**交集面积 ÷ 并集面积**。交集 = 各自面积的"重叠部分"，并集 = 两个面积之和减去重叠（避免重复算）。

```python
def iou_2d(box_a, box_b):
    """两个 xyxy 框的 IoU"""
    # 交集: 各边取"靠内"的值
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)   # 无交集则 0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def iou_matrix(boxes, gt_boxes):
    """批量: (N,4) vs (M,4) -> (N,M) IoU 矩阵（匹配算法的基础）"""
    N, M = boxes.shape[0], gt_boxes.shape[0]
    inter = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            inter[i, j] = iou_2d(boxes[i], gt_boxes[j])
    return inter
```

**旋转框 IoU（面试口述思路即可）**：
1. 旋转框 → 4 个顶点（中心 + w/l + theta 算角点）
2. 求两个多边形的**交集多边形**：Sutherland–Hodgman 裁剪算法（每个边裁剪一次）
3. 交集面积（鞋带公式 shoelace）+ 并集面积 → IoU
4. **3D 近似**：BEV 投影算 2D IoU × z 轴重叠率，或采样点落在框内比例（nuScenes 官方用 BEV IoU）

**面试追问**：
- Q：IoU 和 GIoU/DIoU/CIoU 递进关系？→ 见 [[模型基础知识补充]]（IoU 不重叠无梯度 → GIoU 外包矩形 → DIoU 中心距 → CIoU 长宽比）。
- Q：为什么旋转框 IoU 难算？→ 旋转矩形交点不再轴对齐，需要多边形裁剪，且数值稳定性差（面积接近 0 时）。

---

## 四、Focal Loss 手写 ⭐⭐⭐

**为什么考**：检测/分割标配损失，常要求写前向 + 说 γ 的作用（[[模型基础知识补充]] 已有公式，这里写实现）。

**大白话**：CE 会被海量"简单负样本"（背景）淹没。Focal 加调制因子 `(1-p_t)^γ`：**预测越准的样本权重越小**，让模型把注意力留给难样本。

```python
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    logits: (N,) 未过 sigmoid 的原始输出
    targets: (N,) 0/1 标签
    返回: 标量损失
    """
    p = 1.0 / (1.0 + np.exp(-logits))         # sigmoid
    p_t = np.where(targets == 1, p, 1 - p)    # 对应类别的预测概率
    # 标准 CE: -log(p_t)，数值稳定写法
    ce = -np.log(p_t + 1e-8)                  # 加 eps 防 log(0)
    # 调制因子: (1-p_t)^γ —— 预测准 -> p_t->1 -> 因子->0
    modulating = np.power(1.0 - p_t, gamma)
    # alpha 平衡正负样本（类别加权，可选）
    alpha_t = np.where(targets == 1, alpha, 1 - alpha)
    loss = alpha_t * modulating * ce
    return loss.mean()

# 验证: p_t=0.9 (简单样本) 时 modulating = 0.1² = 0.01 → 贡献压到 1%
#        p_t=0.5 (难样本) 时 modulating = 0.5² = 0.25 → 保留 25%
```

**面试追问**：
- Q：γ=0 时是什么？→ 退化成加权 CE（只剩 alpha 项）。
- Q：为什么 α 和 γ 一起用？→ α 平衡"正负样本数量"，γ 平衡"简单难样本权重"——一个管数量，一个管难度。
- Q：数值稳定性怎么做？→ sigmoid + BCE 用 log-sum-exp 合并，避免 exp 溢出（`p = sigmoid(x)` 与 `ce = log1p(exp(-x))` 等价变形）。

---

## 五、BatchNorm / LayerNorm / GroupNorm 前向 ⭐⭐⭐

**为什么考**：归一化是高频题，常要求写前向 + 说**训练/推理模式区别**。

**大白话**：训练时每个 batch 算均值和方差做归一化，**再学两个可训练参数 γ（缩放）β（平移）**；推理时用训练期累积的 running mean/var（因为单样本没有 batch 统计量）。

```python
def batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
    """
    x: (N, C, H, W) 或 (N, C)
    训练: 用当前 batch 的 mean/var 归一化 + 更新 running 统计
    推理: 直接用 running 统计（固定），不再更新
    """
    is_4d = x.ndim == 4
    if is_4d:
        # gamma/beta/running 需对齐通道维: (C,) -> (1, C, 1, 1) 才能广播
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        running_mean = running_mean.reshape(1, -1, 1, 1)
        running_var = running_var.reshape(1, -1, 1, 1)
    if training:
        axes = (0, 2, 3) if is_4d else 0
        batch_mean = x.mean(axis=axes, keepdims=True)
        batch_var = x.var(axis=axes, keepdims=True)
        # 更新 running 统计（滑动平均），存回 (C,) 形状
        rm = (1 - momentum) * running_mean + momentum * batch_mean
        rv = (1 - momentum) * running_var + momentum * batch_var
        mean, var = batch_mean, batch_var
        running_mean, running_var = (rm.squeeze(), rv.squeeze()) if is_4d else (rm, rv)
    else:
        mean, var = running_mean, running_var      # 推理: 用历史统计
    x_norm = (x - mean) / np.sqrt(var + eps)       # 归一化
    return gamma * x_norm + beta, running_mean, running_var

def layer_norm(x, gamma, beta, eps=1e-5):
    """对每个样本的"所有特征"归一化（Transformer 用）"""
    mean = x.mean(axis=-1, keepdims=True)          # 沿特征维
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def group_norm(x, gamma, beta, num_groups=32, eps=1e-5):
    """把通道分 G 组，组内归一化（BN 依赖 batch 大小，GN 不依赖——单卡小 batch 用）"""
    N, C = x.shape[:2]
    g = x.reshape(N, num_groups, C // num_groups, -1)
    mean = g.mean(axis=(2, 3), keepdims=True)
    var = g.var(axis=(2, 3), keepdims=True)
    g = (g - mean) / np.sqrt(var + eps)
    # gamma/beta 按通道: (C,) -> (1, C, 1, 1) 对齐
    return gamma.reshape(1, C, 1, 1) * g.reshape(x.shape) + beta.reshape(1, C, 1, 1)
```

**面试追问**：
- Q：BN 训练和推理有什么不同？→ 训练用 batch 统计并更新 running；推理用 running（单样本无统计意义），且推理时 BN 可融合进卷积（推理加速，见 [[模型部署与延迟优化]]）。
- Q：Transformer 为什么用 LN 不用 BN？→ 序列长度可变、batch 通常小、token 间差异大；LN 沿特征维归一化，不依赖 batch 大小，训练推理一致。
- Q：GN 解决什么？→ BN 在 batch=1 时统计量是噪声；GN 不依赖 batch，检测/分割大图小 batch 常用（[[Transformer进阶知识]] 中 Swin 等视觉模型标配）。

---

## 六、RoPE（旋转位置编码）⭐⭐

**为什么考**：2023 后大模型/VLA 标配（[[Transformer进阶知识]] 有公式），进阶题。

**大白话**：给每个位置 m 配一个旋转角 `m·θ`，把 Q/K 的每对维度 (2i, 2i+1) **旋转这个角度**。两个 token 的 Q·K 点积后，旋转角相减——**只剩相对位置 (m-n)**，天然编码相对位置，还能外推。

```python
def rope(q, k, seq_len, dim, base=10000.0):
    """
    q, k: (seq_len, dim) 已投影的 query/key
    返回: 旋转后的 q, k（位置信息以"旋转"形式注入）
    """
    # 1. 每个维度的频率: θ_i = base^(-2i/dim)
    freqs = 1.0 / (base ** (np.arange(0, dim, 2) / dim))   # (dim/2,)
    # 2. 每个位置的旋转角: m·θ_i
    m = np.arange(seq_len)[:, None]                        # (seq_len, 1)
    angles = m * freqs[None, :]                            # (seq_len, dim/2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    # 3. 把 (dim) 分成相邻对 (2i, 2i+1)，做二维旋转:
    #    [x0] -> [cos, -sin] [x0]    对每对 (x_{2i}, x_{2i+1})
    #    [x1]    [sin,  cos] [x1]
    def rotate(x):
        x = x.reshape(seq_len, -1, 2)                      # 拆成对
        x0, x1 = x[..., 0], x[..., 1]
        # 旋转后: 新0 = x0·cos - x1·sin, 新1 = x0·sin + x1·cos
        rot = np.stack([x0 * cos - x1 * sin,
                        x0 * sin + x1 * cos], axis=-1)
        return rot.reshape(seq_len, -1)
    return rotate(q), rotate(k)
# 关键性质: q_m · k_n 只依赖 (m-n)（旋转角相减），即"相对位置天然编码"
```

**面试追问**：
- Q：RoPE 和绝对位置编码区别？→ 绝对编码把位置加进 embedding（位置信息是"绝对序号"）；RoPE 旋转进 Q/K，点积后只剩相对位置，且能外推到更长序列（通过缩放 θ）。
- Q：为什么能外推？→ 旋转角只依赖位置差，训练见过的"相对距离"范围可以泛化；超长外推用 NTK/插值技巧。

---

## ✅ 自测一下（先想答案，再点开）

> [!question] Q1：手写 Self-Attention，说出 O(N²) 空间花在哪、FlashAttention 怎么省？

> [!success]- 参考答案
> 空间花在 attention 矩阵 (N,N) 的存储。FlashAttention 分块（tiling）把 Q/K/V 分成小块，在 SRAM 里算完一个 block 的 softmax（Online Softmax）直接输出，不落 HBM；反向重算不缓存。空间 O(N²)→O(N)，且是精确计算（[[Transformer进阶知识]] 一节的实现细节）。

> [!question] Q2：NMS 和 Soft-NMS 的取舍？什么场景 Soft-NMS 明显更好？

> [!success]- 参考答案
> NMS 直接删 IoU 超阈值的框——密集场景两个真目标重叠会误删低分真框（如两人并排、车群）。Soft-NMS 按 IoU 衰减分数（高斯/线性），高重叠但分高的框保留、分低的降到阈值以下再删。密集行人/车辆场景收益明显；代价是多了超参（sigma）和少量计算。

> [!question] Q3：BN 推理时为什么不用 batch 统计？

> [!success]- 参考答案
> 推理常是 batch=1（或很小的 batch），batch 统计量噪声大且和训练分布不一致；而且推理要求"单样本结果确定"。所以训练期用滑动平均维护 running mean/var，推理固定使用，还能与卷积融合省一次归一化运算（[[模型部署与延迟优化]]）。

> [!question] Q4：Focal Loss 里 p_t=0.9、γ=2 的样本损失被压到多少？为什么？

> [!success]- 参考答案
> 调制因子 (1-0.9)² = 0.01，损失压到原来的 1%。因为 p_t=0.9 是"简单样本"（模型已很确定），Focal 想让它少贡献梯度，把学习资源留给 p_t≈0.5 的难样本。γ 越大压得越狠，γ=0 退化为 CE。

---

## 🛠 动手练习（面试前 2 周，每天 1 类）

1. **第 1 天**：不看笔记手写 Attention，口述复杂度；再写 Masked Attention（因果 mask）。
2. **第 2 天**：手写 NMS + iou_matrix；用随机框验证 `keep` 数量随阈值变化（0.3/0.5/0.7）。
3. **第 3 天**：写 focal_loss，用 `p=[0.9, 0.5, 0.1]` 手算一遍对答案（[[模型基础知识补充]] 练习 1 有答案）。
4. **第 4 天**：写 BN 前向（训练/推理分支），再写 LN；说出三个归一化各自沿哪一维。
5. **第 5 天**：写 RoPE 的 cos/sin 表构造，口述"为什么点积只剩相对位置"。
6. **第 6 天**：把 6 类算法各用**一句话**写出推导链（如 "softmax 减 max 防溢出 → 缩放 √d → 加权 V"），写进自己的面试复盘（可参考 [[查漏补缺与求职行动清单]] ➕ 项规划一个 [[面试复盘记录]]）。
7. **进阶**：3D NMS / 旋转框 IoU 用多边形裁剪实现（口头流程即可，很少要求完整写）。

---

## ➡️ 下一步学什么

1. **[[查漏补缺与求职行动清单]]** —— 🔥 第 1 项完成，接着补第 2 项（mmdet3d 源码走读）。
2. **[[常见面试题-感知算法]]** —— Q10-Q13（Attention 复杂度/Deformable/Flash）直接复用本篇实现。
3. **[[智驾算法面试题库]]** —— Q21 匈牙利算法（匹配手撕）是第 7 类高频手写题。
4. **[[模型部署与延迟优化]]** —— BN 融合卷积、量化对 NMS 的影响（工程追问）。
5. **[[Transformer进阶知识]]** —— RoPE/FlashAttention/GQA 的实现视角（本篇六/一节的深化）。

> 💡 手撕题的隐藏考点：**数值稳定性**（减 max、加 eps、log-sum-exp）。面试官常问"你的代码在极端输入下会崩吗"——把这三处写进代码，比背答案加分。

---

## 相关笔记

- [[面试准备策略]] — 六节手写清单来源 + 12 周计划
- [[查漏补缺与求职行动清单]] — 🔥 第 1 项（本篇是其落地）
- [[模型基础知识补充]] — Focal Loss / 归一化公式与直觉
- [[Transformer架构详解]] — Attention 原理
- [[Transformer进阶知识]] — FlashAttention / RoPE / GQA
- [[智驾算法面试题库]] — Q21 匈牙利匹配手撕
- [[常见面试题-感知算法]] — 算法类面试题
