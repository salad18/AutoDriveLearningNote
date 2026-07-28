---
tags: [training, infrastructure, gpu, distributed, profiling]
created: "2026-07-28"
---

# AI Infra 详解

> 分布式训练基础设施、GPU 优化和性能分析。从单卡训练到千卡集群的工程实践。

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

---

## 四、显存优化技巧

### Gradient Checkpointing

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

### Gradient Accumulation

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

---

> 📚 **延伸阅读**: [[三阶段训练范式]], [[模型训练与微调]], [[数据闭环总览]], [[本地训练方案]]
