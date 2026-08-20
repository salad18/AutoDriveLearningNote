---
tags: [training, infrastructure, gpu, distributed, profiling]
created: "2026-07-28"
updated: "2026-08-21"
---

# AI Infra 详解

> 一句话导读：模型**训得动、训得快**靠的不是"好模型"，是**基础设施**——分布式训练怎么拆、显存怎么省、混合精度怎么选、卡在瓶颈怎么查。本篇从单卡到千卡集群，把"训练工程"的底层逻辑讲透。

---

## 🧭 本篇导读

| 项目 | 内容 |
|------|------|
| **这篇学什么** | 并行策略（DP/TP/PP/EP）、DDP 与通信原语、FSDP/ZeRO 显存切分、混合精度（FP16/BF16）、显存优化、数据加载、Profiling、自动驾驶特有训练挑战、集群最佳实践 |
| **需要的前置知识** | [[模型基础知识补充]]（训练基础）、[[三阶段训练范式]]（训练流程）、PyTorch 基础 |
| **学完之后你能** | ① 说清"模型太大放不下"时该用哪种并行；② 估算任意模型的显存需求并定位"显存炸弹"；③ 用 profiler 找到训练瓶颈；④ 看懂生产环境训练脚本的每个配置项 |
| **预计阅读时间** | 120-150 分钟（信息密度高，建议分 2-3 次读） |

> [!tip] 怎么读这篇
> 这是"工具型"笔记，和 [[模型基础知识补充]] 一样：**第一遍通读建框架，训练时回来查**。重点吃透三个概念：**并行拆法（怎么拆）、显存构成（什么在占显存）、精度取舍（FP16/BF16 怎么选）**。

---

## 〇、大白话总览

### 训练跑不起来，先问三个问题

```
问题 1: 模型太大，单卡放不下？   → 并行策略（TP/PP/FSDP/ZeRO）
问题 2: 显存够但 OOM？          → 显存优化（checkpoint/累积/降分辨率）
问题 3: 卡了很多但没变快？       → Profiling（找瓶颈：数据加载/通信/计算）
```

**一篇笔记解决三件事**：怎么拆（并行）、怎么省（显存）、怎么查（profiling）。

> [!note] 打个比方
> - **并行** = 一栋大楼很多人盖：数据并行（每人盖一层楼的不同房间）vs 张量并行（每人都来砌同一面墙的不同段）vs 流水线并行（按楼层分工）。
> - **显存** = 施工场地的"材料堆放区"：参数（砖）、梯度（图纸）、优化器状态（工具）、激活值（脚手架）——工程队必须盘算每种材料占多少地方。
> - **Profiling** = 监理的"时间表"：谁在等谁（GPU 等数据？等通信？）——找到拖后腿的环节。

---

## 一、分布式训练基础

### 并行策略总览

```
训练并行金字塔:

        ┌──────────────┐
        │  Data Parallel│  ← 数据拆分到多 GPU
        ├──────────────┤
        │Tensor Parallel│  ← 模型层内拆分 (如单层矩阵分块)
        ├──────────────┤
        │Pipeline Paral │  ← 模型层间拆分 (不同层在不同 GPU)
        ├──────────────┤
        │  Expert Paral │  ← MoE 中 Expert 分布到不同 GPU
        └──────────────┘
```

> [!note] 四种并行的直觉（面试必考）
> - **数据并行（DP）**：每个 GPU 一份完整模型，各吃各的数据，梯度汇总。最常用——**模型能放进单卡时首选**。
> - **张量并行（TP）**：一层网络拆成几块分给几个 GPU（比如矩阵乘法按行分块）。**单层都放不下时用**（大模型标配）。
> - **流水线并行（PP）**：网络按层分组，GPU1 跑第 1-10 层、GPU2 跑 11-20 层……**像工厂流水线**（缺点是流水线气泡）。
> - **专家并行（EP）**：MoE 模型的每个"专家"放不同 GPU，token 按路由去找专家。
> **组合拳**：大模型常用 3D 并行（DP+TP+PP），再加 ZeRO/FSDP。

### DataParallel (DP) vs DistributedDataParallel (DDP)

| 维度 | DP | DDP |
|------|-----|------|
| **实现方式** | `nn.DataParallel` | `nn.parallel.DistributedDataParallel` |
| **通信** | 单进程多线程 | 多进程 (每个 GPU 一个进程) |
| **GIL 限制** | 受 Python GIL 限制 | 不受 GIL 限制 |
| **负载均衡** | GPU0 负责 gather + broadcast (瓶颈) | 每 GPU 独立 all-reduce |
| **通信效率** | 低 | 高 (NCCL 后端) |
| **使用** | 已废弃 | 生产标准 |

```python
# DDP 使用模板
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 在每个进程:
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

> [!warning] 面试高频：为什么 DDP 替代了 DP？
> DP 是"单进程多线程"，受 GIL 限制且 GPU0 承担所有 gather/broadcast（通信瓶颈、负载不均衡）。DDP 是"每 GPU 一个独立进程"，不受 GIL 限制，梯度用高效的 Ring All-Reduce 在每个 GPU 上直接同步——**又快又均衡**。记住一句话：**生产环境永远用 DDP，不用 DP。**

### All-Reduce 通信原语

```
Ring All-Reduce:

GPU0 ──→ GPU1 ──→ GPU2 ──→ GPU3
  ↑                          │
  └──────────────────────────┘

第 1 步 (Scatter-Reduce): 每个 GPU 发送 1/N 数据，环形传递，各 GPU 收到 N 个部分 reduction
第 2 步 (All-Gather): 环形传递 reduction 结果，所有 GPU 得到完整结果

通信量: 2(N-1)/N × data_size ≈ 2 × data_size (与 GPU 数量无关!)
```

**常用通信原语**:

| 原语 | 功能 | 复杂度 |
|------|------|--------|
| `all_reduce` | 所有 GPU 对 tensor 做 reduce (sum/avg)→所有 GPU 得到结果 | 2(N-1)/N × data |
| `all_gather` | 每 GPU 的 tensor 收集到所有 GPU | (N-1) × data |
| `reduce_scatter` | reduce → 分片到各 GPU | (N-1)/N × data |
| `broadcast` | GPU0 发送到所有 GPU | log(N) 级 |

> [!note] Ring All-Reduce 的"惊人结论"
> 通信总量 ≈ **2 × 数据量，与 GPU 数量无关**！直觉：环形拓扑让每个 GPU 只和自己的邻居通信，数据像"击鼓传花"传一圈，GPU 越多，每个 GPU 承担的分片越小——**总通信量不随规模增长**（这就是 DDP 能线性扩展的底层原因）。

---

## 二、FSDP & ZeRO 详解

### ZeRO 三阶段

```
假设模型参数 P = 7.5B, GPU=8, FP16:

标准训练 (DDP):
  参数: 7.5B × 2 bytes = 15 GB
  梯度: 7.5B × 2 bytes = 15 GB
  Optimizer (Adam): 7.5B × 12 bytes = 90 GB  (fp32 param + momentum + variance)
  激活值: ~10 GB (依赖 batch size)
  总计: ~130 GB/GPU → 需要 A100 80GB × 2 = 不可能

ZeRO-1 (Optimizer State Sharding):
  每个 GPU: 15 + 15 + 90/8 + 10 = 51.25 GB

ZeRO-2 (+ Gradient Sharding):
  每个 GPU: 15 + 15/8 + 90/8 + 10 = 28.125 GB

ZeRO-3 (+ Parameter Sharding):
  每个 GPU: 15/8 + 15/8 + 90/8 + 10 = 25 GB  (8 GPU 内)
  扩展到 64 GPU: 15/64 + 15/64 + 90/64 + 10 ≈ 11.9 GB
```

> [!warning] 这张表要"读懂算账"，不要背数字
> **显存的四块构成**：参数 + 梯度 + 优化器状态 + 激活值。
> - **优化器状态最肥**（Adam 每个参数要存 3 份 FP32：参数副本 + momentum + variance = 12 字节/参数，占 90GB！）
> - ZeRO 的思路：**反正多 GPU 要同步，不如"每块只存一份"**——状态分散在各 GPU（ZeRO-1 分优化器 → ZeRO-2 分梯度 → ZeRO-3 连参数也分）。
> - **ZeRO-3 的代价**：参数用的时候要"取回来"（通信），所以 ZeRO-3 通信量最大——**省显存换通信**，要权衡。

### FSDP vs DeepSpeed ZeRO 对比

| 维度 | PyTorch FSDP | DeepSpeed ZeRO |
|------|-------------|----------------|
| **生态** | PyTorch 原生 | Microsoft 维护 |
| **易用性** | 较简单 (`wrap` policy) | 配置灵活但复杂 |
| **CPU Offload** | 支持 | ZeRO-Infinity (更强) |
| **通信优化** | 基础 | 激进 (通信/计算 overlap) |
| **社区** | PyTorch 官方 | 大模型训练首选 |

```python
# FSDP 基本用法
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision=MixedPrecision(param_dtype=torch.float16)
)
```

> [!note] 选型建议
> **PyTorch 生态用 FSDP（原生、简单），追求极致性能用 DeepSpeed ZeRO（通信优化更激进）**。规模不大（<13B）时 FSDP 足够，超大模型（100B+）倾向 DeepSpeed + 3D 并行。

---

## 三、混合精度训练 (AMP)

### FP16/FP32/BF16 对比

| 格式 | 总位数 | 指数位 | 尾数位 | 动态范围 | 精度 | 硬件 |
|------|--------|--------|--------|----------|------|------|
| **FP32** | 32 | 8 | 23 | ~10^-38 ~ 10^38 | 最高 | 通用 |
| **FP16** | 16 | 5 | 10 | ~6e-8 ~ 65504 | 中 | V100, A100 |
| **BF16** | 16 | 8 | 7 | ~10^-38 ~ 10^38 | 低 (但范围=FP32) | A100, H100 |
| **TF32** | 19 | 8 | 10 | ~FP32 范围 | 近 FP32 | A100, H100 |

**BF16 优势**：
- 动态范围 = FP32 → 不需要 loss scaling
- 但精度低于 FP16 (尾数只有 7 位)
- 适合大模型训练 (overflow 比精度损失更致命)

> [!note] 一句话选型（面试高频）
> - **FP16**：精度高但范围窄（最大 65504），需要 loss scaling 防溢出——适合感知模型。
> - **BF16**：范围 = FP32（不会溢出），精度略低——**适合大模型**（大模型 overflow 更致命，且 BF16 不需要 loss scaling 更省心）。
> - **记忆点**：FP16"怕溢出"（小范围），BF16"怕不精确"（小尾数）——**选谁取决于"溢出"和"不精确"哪个更致命**。

### AMP 完整流程

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()  # loss scaling

for data, target in dataloader:
    optimizer.zero_grad()

    # Forward: 自动选择 FP16/BF16
    with autocast(device_type='cuda', dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    # Backward: 使用 scaled loss
    scaler.scale(loss).backward()

    # Unscale + Update
    scaler.step(optimizer)
    scaler.update()  # 动态调整 scale

# GradScaler 内部逻辑:
# 1. loss × scale_factor → FP16 backward
# 2. 检查梯度是否有 Inf/NaN
# 3. 如果有 → 跳过此次更新 + 减小 scale_factor
# 4. 如果没有 → unscale grad + optimizer.step + 增大 scale_factor
```

> [!note] loss scaling 在防什么？
> FP16 下梯度值可能小于 6e-8（下溢成 0）——**梯度消失**。GradScaler 先把 loss 放大（×scale）再反向，梯度也相应放大，就不会下溢；更新前再缩小。**检查 Inf/NaN 是核心**：梯度爆了就跳过一次更新并调小 scale——这就是"动态缩放"的自我保护。

### 自动驾驶训练中的精度策略

```
感知模型 (BEV/3D Detection):
  - 推荐: FP16 + AMP → 绝大多数层安全
  - 注意: Depth estimation 层可能敏感 → 保持 FP32
  - Loss 中涉及 exp() 或 large dynamic range 的保持 FP32

大模型 (LLM in Driving, VLMs):
  - 推荐: BF16 (不需要 loss scaling)
  - FSDP + BF16 → 最优训练方案
```

> [!warning] "敏感层保持 FP32"是实用技巧
> 不是所有层都适合低精度：**深度估计层**（softmax 分布、范围大）和**含 exp() 的 loss** 对精度敏感。做法：用 `amp.register_float_function` 或手动把敏感模块切成 FP32（[[模型部署与延迟优化]] 的 INT8 逐层分析是同一思路的部署版）。

---

## 四、显存优化技巧

### Gradient Checkpointing（时间换空间）

```python
# 原理: 不在 forward 时存所有激活值，backward 时重新计算
# 时间换空间

from torch.utils.checkpoint import checkpoint

class BEVEncoder(nn.Module):
    def forward(self, x):
        # 标记哪些 block 不存激活值
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
        return x

# 显存节省: ~30-50% (取决于 checkpoint 的层数)
# 时间增加: ~15-25% (额外 forward pass)
```

> [!note] 直觉：不存"脚手架"，用时重搭
> 反向传播需要 forward 的中间结果（激活值）。Checkpoint 的做法：**不存这些激活值，backward 时重新 forward 一次算出来**。省 30-50% 显存，代价是多一次 forward（+15-25% 时间）。**显存紧张时是最快生效的开关。**

### Gradient Accumulation（小 batch 攒大 batch）

```python
# 场景: 每 GPU 只能跑 batch_size=2，但需要 effective bs=64
# 解法: 梯度累积

accumulation_steps = 32  # 使 effective_bs = 2 × 32 = 64 (单 GPU)
optimizer.zero_grad()
total_loss = 0

for i, batch in enumerate(dataloader):
    # 关键: 累积的 loss 要归一化!
    loss = model(batch) / accumulation_steps
    loss.backward()
    total_loss += loss.item()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 注意: BatchNorm 在 accumulation 时行为有变化
# BN 在每个 micro-batch 独立计算 stats → 全局 BN 用 SyncBatchNorm
```

> [!warning] 两个容易踩的坑
> ① **loss 要除以 accumulation_steps**：否则累积的梯度是"32 个 batch 的和"，等于把学习率放大了 32 倍；② **BatchNorm 的行为**：BN 按 micro-batch 独立算统计量，accumulation 不能模拟大 batch 的 BN——**多卡用 SyncBatchNorm 同步统计量**。

### 显存预算估算

```
训练一个 BEV 模型的显存计算:

模型参数: 100M × 4 bytes (FP32 master) = 400 MB
         + 100M × 2 bytes (FP16 copy) = 200 MB
梯度: 100M × 2 bytes = 200 MB
优化器: 100M × 12 bytes (Adam FP32) = 1200 MB

激活值 (最难估算):
  = batch_size × resolution × channels × n_layers × bytes_per_element
  例: bs=4, BEV=128×128×256维, 6层Transformer
  ≈ 4 × 128×128 × 256 × 6 × 2 bytes ≈ 200 MB

数据加载: ~50-200 MB (图像+点云)

总计 (DDP): ~2.2 GB + 激活值 (随 batch_size 线性增长)
总计 (FSDP/8GPU): ~275 MB + 激活值 (基本不变)
```

> [!note] 估算的价值：**先算后调**
> 看到"OOM"，先用这套公式估算：是参数/优化器（换 FSDP）还是激活值（降 batch/checkpoint）还是数据（降 num_workers）？**算一遍就知道该动哪里，而不是瞎试**。

---

## 五、数据加载优化

### DataLoader 瓶颈诊断

```python
# 常见瓶颈: CPU 数据预处理跟不上 GPU 计算

# 优化手段:
dataloader = DataLoader(
    dataset,
    batch_size=bs,
    num_workers=8,          # 增加 worker 数 (通常 CPU 核心数)
    pin_memory=True,        # 使用锁页内存加速 CPU→GPU 传输
    prefetch_factor=4,      # 每个 worker 预取 batch 数
    persistent_workers=True  # worker 不随 epoch 重启
)

# 多模态数据特殊优化:
# 1. 图像解码在 GPU 进行 (DALI)
# 2. 点云 voxelization 预计算缓存
# 3. 使用 mmap 加载大文件 (nuScenes .bin)
```

> [!warning] GPU 利用率低的头号嫌疑：数据加载
> 症状：GPU 利用率 <50%，且 GPU 在"等待"。**排查顺序**：num_workers 够吗 → pin_memory 开了吗 → 预处理在 CPU 还是 GPU → 是不是反复解码同一批图像。数据加载优化通常比模型优化性价比高得多。

### DALI (NVIDIA Data Loading Library)

```
传统 pipeline:
  CPU: 读取 → 解码 → 预处理 → → GPU: 训练
  ↑ 瓶颈: JPEG 解码 + resize 在 CPU 慢

DALI pipeline:
  CPU: 读取 → → GPU: 解码 → 预处理 → 训练
  ↑ GPU 专门硬件解码 JPEG，10-50x 快于 CPU

使用时机:
  - 当 CPU utilization 接近 100% 但 GPU 在等待
  - 大分辨率图像 (1920×1080+)
  - 多视角图像 (6-8 cameras)
```

> [!note] 自动驾驶为什么特别需要 DALI？
> 6-8 个相机、每帧 1600×900+、JPEG 解码在 CPU 上是重活——**多相机大图是 CPU 解码的噩梦**。DALI 把解码搬到 GPU（硬件解码器），10-50 倍提速。**"CPU 100% 但 GPU 空转"时，DALI 是标准答案。**

---

## 六、Profiling 与性能分析

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()

# 在 TensorBoard 中查看: tensorboard --logdir=./logs
# 关键指标:
# - GPU Utilization: 应该 > 80%
# - Kernel Launch Overhead: 应该 < 5% 总时间
# - Memory Bandwidth Utilization: 计算密集型 vs 访存密集型
```

### 常见性能瓶颈

| 症状 | 可能原因 | 解法 |
|------|----------|------|
| GPU 利用率 < 50% | DataLoader 太慢 | 增加 num_workers, 用 DALI |
| GPU 利用率间歇性 100%→0% | 通信阻塞 | 通信/计算 overlap (gradient bucketing) |
| 显存 OOM 但利用率低 | 碎片化 | 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 训练速度不随 GPU 增加线性提升 | 通信瓶颈 | 增大 batch size, 用 gradient accumulation |
| 同 batch 速度波动大 | 数据形状不一致 | 固定输入尺寸, 避免 dynamic shape |

> [!note] 读表的方法：**症状 → 病因 → 解法**是排查模板
> 每个症状都对应一个"最常见病因"：利用率低=数据慢、间歇 100%→0%=通信、OOM 但利用率低=碎片化、扩展不线性=通信占比高、速度波动=动态形状。**面试被问"训练慢怎么排查"，按这张表回答就是完整答案。**

### nsys (NVIDIA Nsight Systems)

```bash
# 端到端 profiling
nsys profile -o output.qdrep python train.py

# 关键分析:
# - CUDA API trace: kernel launch 时间线
# - GPU Operations: 每个 kernel 的时长
# - CPU/GPU 同步: 哪些操作在等待 GPU
# - Memory: H2D/D2H 传输量
```

---

## 七、自动驾驶特有的训练挑战

### BEV 模型训练显存优化

```
挑战: BEV 网格 × 6-8 相机 = 大显存需求

优化方案:
1. 降低 BEV 分辨率: 256×256 → 128×128 → 80×80
2. 时序帧采样: 不存所有历史帧的激活值 → gradient checkpoint
3. 多相机分开处理: 逐相机 forward → 聚合 BEV
4. Deformable Attention: O(NK) 替代 O(N²)
5. 深度估计网络压缩: BEVDepth 的 depth net 用轻量架构
```

### 多任务联合训练的梯度管理

```python
# 自动驾驶常见多任务: 检测 + 分割 + Map + Motion + Planning

# 问题: 各 task 的 loss magnitude 不同
# 检测 loss: ~1-5
# 分割 loss: ~0.1-0.5
# 规划 loss: ~0.01-0.1

# 解法 1: 手动调权重 (最常用但费事)
total_loss = 1.0 * det_loss + 5.0 * seg_loss + 10.0 * plan_loss

# 解法 2: Uncertainty Weighting (自动学习)
# loss = Σ 1/(2σ²) * L_i + log(σ)  → σ 是可学习参数

# 解法 3: GradNorm (动态平衡学习速度)
# 监控每个 task 的 gradient norm 和学习速度，实时调整权重
```

> [!warning] 多任务 loss 失衡是 [[UniAD详解]] 分阶段训练的根本原因之一
> 检测 loss ~5、规划 loss ~0.01（差 500 倍），直接加总会被检测主导。解法：手动权重（简单但费事）、Uncertainty Weighting（自动学 σ）、GradNorm（按梯度平衡）。**UniAD 用"分阶段训练"绕开这个问题**——训练工程和模型设计是互相影响的。

### 长尾/Corner Case 训练

```
自动驾驶的核心难点: 99.9% 的常见场景 + 0.1% 的致命 corner case

训练策略:
1. Focal Loss: 自动降低简单样本权重
   FL(p) = -(1-p)^γ * log(p), γ=2

2. Hard Example Mining (OHEM):
   - 每个 batch 只反向传播 loss 最高的 K 个样本

3. 数据重采样:
   - 稀有场景 (夜间/雨雪) 的采样概率提高 5-10x

4. Active Learning:
   - 从大量未标注数据中选出模型最不确定的样本 → 人工标注 → 加入训练
```

> [!note] 和 [[数据闭环总览]] 的联动
> 这里的"长尾训练策略"是**训练侧**的应对（Focal/OHEM/重采样/主动学习），数据闭环是**数据侧**的应对（采集挖掘 corner case）——**双侧并行**才是完整打法。

---

## 八、集群训练最佳实践

### 训练超参速查表

| 场景 | Batch Size | LR | Warmup | Optimizer | Precision |
|------|-----------|-----|--------|-----------|-----------|
| BEVDet (8 GPU) | 32-64 | 2e-4 | 500 steps | AdamW | FP16 |
| BEVFormer (8 GPU) | 8-16 | 2e-4 | 1000 steps | AdamW | FP16 |
| CenterPoint (4 GPU) | 16-32 | 0.01 | 1000 steps | SGD+Momentum | FP32 |
| UniAD (8 GPU) | 4-8 | 1e-4 | 2000 steps | AdamW | FP16 |
| LLM (64 GPU) | 128-512 | 3e-4 | 2000 steps | AdamW(0.9,0.95) | BF16 |

### 训练脚本模板

```python
# 完整的多节点训练脚本结构
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main_worker(rank, world_size):
    # 1. 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2. 模型 (可选 FSDP wrap)
    model = build_model().cuda()
    model = DDP(model, device_ids=[rank])

    # 3. 数据 (DistributedSampler 确保不重复)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, ...)

    # 4. 优化器 + AMP
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = GradScaler()

    # 5. 训练循环
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 每个 epoch 不同 shuffle
        for batch in dataloader:
            with autocast():
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    dist.destroy_process_group()

# 启动命令:
# torchrun --nproc_per_node=8 --nnodes=2 train.py
```

> [!warning] 新手最容易漏的两个细节
> ① **`sampler.set_epoch(epoch)`**：不设置的话每个 epoch 的 shuffle 顺序一样，模型"背数据"；② **`torch.cuda.set_device(rank)`**：不设置会用默认 GPU0，多卡全挤在一张卡上。

---

## 九、常见误区速查

> [!warning] 新手最常踩的坑
> 1. **以为 FSDP 是"免费的显存"**——省显存换通信，ZeRO-3 通信量最大，小模型用 FSDP 反而可能变慢。
> 2. **混淆 FP16 和 BF16 的取舍**——FP16 怕溢出（范围小），BF16 怕不精确（尾数少）；大模型选 BF16 不是因为"精度高"，是因为"不会溢出且省 loss scaling"。
> 3. **梯度累积忘记除 accumulation_steps**——等于悄悄把学习率放大 N 倍，训练直接崩。
> 4. **以为多卡 = 自动线性加速**——通信占比随卡数上升，小 batch 卡多了通信反而拖慢（扩展性要先算通信/计算比）。
> 5. **GPU 利用率低就优化模型**——先查数据加载（num_workers/pin_memory/DALI），数据瓶颈是最常见也最好修的问题。

---

## ✅ 检验自己（自测题）

> [!question] Q1：四种并行策略（DP/TP/PP/EP）分别在什么场景用？
> 提示：数据 / 单层 / 层间 / 专家。

> [!success]- 参考答案
> 数据并行（DP/DDP）：模型能放进单卡，加速训练（最常用）。张量并行（TP）：单层网络都放不下时，把一层拆到多卡（大模型标配）。流水线并行（PP）：模型按层分组放不同卡（像工厂流水线，有气泡开销）。专家并行（EP）：MoE 模型的专家分布到不同卡。大模型实际是组合拳：3D 并行（DP+TP+PP）+ ZeRO/FSDP。

> [!question] Q2：ZeRO 三阶段分别切分什么？为什么优化器状态最肥？
> 提示：优化器 / 梯度 / 参数。

> [!success]- 参考答案
> ZeRO-1 切优化器状态、ZeRO-2 加切梯度、ZeRO-3 连参数也切。优化器状态最肥：Adam 每个参数要存 3 份 FP32（参数副本 + momentum + variance = 12 字节/参数），7.5B 模型就是 90GB——比参数本身（15GB）和梯度（15GB）大 6 倍。ZeRO 本质是"多 GPU 反正要同步，不如分着存"，代价是通信量上升。

> [!question] Q3：FP16 和 BF16 各自怕什么？自动驾驶感知模型和大模型分别怎么选？
> 提示：溢出 vs 不精确。

> [!success]- 参考答案
> FP16 范围小（最大 65504）怕梯度/值溢出（需要 loss scaling 动态防溢出）；BF16 尾数少怕不精确（但范围=FP32 不会溢出）。感知模型（BEV/3D 检测）用 FP16+AMP，深度估计等敏感层保持 FP32；大模型（VLM/LLM）用 BF16（不需要 loss scaling，overflow 比精度损失更致命）。选型逻辑：哪个问题更致命就避开哪个。

> [!question] Q4：Gradient Checkpointing 和 Gradient Accumulation 各是"拿什么换什么"？
> 提示：时间换空间 / 时间换 batch。

> [!success]- 参考答案
> Gradient Checkpointing：不存激活值，backward 时重新 forward 计算——**时间换空间**（省 30-50% 显存，+15-25% 时间）。Gradient Accumulation：多个 micro-batch 攒够梯度再更新——**时间换大 batch**（模拟大 batch，但 BN 统计量仍按 micro-batch 算）。注意：累积时 loss 要除以 accumulation_steps，BN 多卡用 SyncBatchNorm。

> [!question] Q5：GPU 利用率 <50%，你的排查顺序是什么？
> 提示：数据 → 通信 → 计算。

> [!success]- 参考答案
> ① 先查数据加载：num_workers 是否够、pin_memory 是否开、是否有 CPU 解码瓶颈（多相机大图用 DALI）；② 再查通信：利用率是否间歇性 100%→0%（通信阻塞，用 gradient bucketing overlap）；③ 最后查计算：kernel launch overhead、显存碎片（expandable_segments）。经验：数据瓶颈最常见也最好修，先查数据再动模型。

---

## 🛠 动手练习

### 练习 1：显存估算实战（30 分钟）

给定一个 3B 参数模型（约等于 30 亿参数），FP16 训练，8 卡，bs=2/卡：
1. 参数 / 梯度 / 优化器状态各多少 GB？（用本文公式：参数 2B/参、梯度 2B/参、Adam 12B/参）
2. DDP 下单卡显存多少？（激活值暂按 5GB 估）
3. 换成 FSDP（ZeRO-3）后单卡多少？
4. 结论：这张卡要多少显存才够？（A100 80GB 够吗？）

> [!tip] 算完你会理解
> 3B 模型 DDP ≈ 3B×16B ≈ 48GB + 激活 5GB ≈ 53GB（80GB 卡勉强）；FSDP 后 ≈ 3B×16B/8 ≈ 6GB + 5GB ≈ 11GB（随便跑）。**这就是为什么大模型必须 FSDP。**

### 练习 2：写一个 DDP + AMP + 梯度累积的训练脚本（60-90 分钟）

基于第八节的脚本模板，实现一个包含以下功能的完整脚本：
1. DDP 初始化（多进程）。
2. AMP（autocast + GradScaler）。
3. 梯度累积（accumulation_steps=8，loss 归一化）。
4. `sampler.set_epoch(epoch)` 正确设置。
5. 用 `torchrun --nproc_per_node=2` 在本地两张卡（或 CPU fallback 用 gloo）跑通 MNIST 小模型。

> [!tip] 做完后自问
> ① 去掉 `loss/accumulation_steps` 会发生什么？（loss 曲线异常）② 去掉 `set_epoch` 会发生什么？（每 epoch 数据顺序一样）③ 把 `gradient_clip` 加上，max_norm 设多少合理？（1.0）

### 练习 3：Profiling 实战（可选，1-2 小时）

用 PyTorch Profiler 分析你训练脚本的一个小模型（如 [[计算机视觉基础]] 的 TinyCNN on MNIST）：
1. 生成 trace 并在 TensorBoard 查看。
2. 找出：GPU 利用率、最耗时的 3 个 kernel、CPU 等待 GPU 的时间。
3. 模拟"数据加载慢"：把 num_workers 设为 0 再跑一次，对比 GPU 利用率的变化。
4. 写下你的结论：这个训练的瓶颈在哪？

> [!tip] 这是"训练工程"的入门仪式
> 亲手看一次 trace，比读十篇 profiling 教程都有效。把发现写进 [[2026-08-20]]。

---

## ➡️ 下一步学什么

按知识库学习路径，读完本篇你应该接着：

1. **[[模型部署与延迟优化]]** —— 训练之后是部署：TensorRT、INT8 量化、延迟预算（第 2 批收官篇）。
2. **[[模型训练与微调]]** —— LoRA、灾难性遗忘、联合训练等训练策略细节。
3. **[[本地训练方案]]** —— 想动手：本地 CPU + DirectML GPU 环境怎么搭。
4. **[[训练排错实战手册]]** —— 训练出问题时的系统排查流程。

> 💡 第 2 批第三篇完成！下一篇 [[模型部署与延迟优化]]——"训完怎么装上车、怎么快"。

---

## 相关笔记

- [[三阶段训练范式]] — 现代训练流程
- [[模型训练与微调]] — 训练策略细节
- [[模型部署与延迟优化]] — 部署与推理优化
- [[数据闭环总览]] — 训练数据生产
- [[本地训练方案]] — 本地动手训练
- [[训练排错实战手册]] — 训练问题排查
- [[模型基础知识补充]] — 训练基础
