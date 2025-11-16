# 全局性能优化：专家级方法论

> **核心思想**：单个 kernel 优化是战术，全局优化是战略。90% 的工程师只做战术优化，但真正的 10x 性能提升来自战略层面。

---

## 目录

1. [性能分析的三个层次](#性能分析的三个层次)
2. [Roofline 模型：性能的物理极限](#roofline-模型性能的物理极限)
3. [Amdahl's Law：优化的投资回报率](#amdahls-law优化的投资回报率)
4. [系统级瓶颈分析框架](#系统级瓶颈分析框架)
5. [实战案例：Transformer 训练优化](#实战案例transformer-训练优化)
6. [反直觉的优化策略](#反直觉的优化策略)
7. [Production 环境的隐藏陷阱](#production-环境的隐藏陷阱)

---

## 性能分析的三个层次

### Level 1: Kernel 层（90% 工程师停留在这里）

```
问题：这个 kernel 为什么慢？
工具：NCU
优化：内存访问、occupancy、ILP
提升：1.2x - 3x
```

**陷阱**：优化了一个占总时间 5% 的 kernel，提升 2x，总体只快了 2.5%

### Level 2: Pipeline 层（优秀工程师的领域）

```
问题：kernels 之间的交互有什么问题？
工具：Nsight Systems
优化：算子融合、并行、重叠
提升：2x - 10x
```

**关键洞察**：
```
总时间 = Σ(kernel_time) + Σ(launch_overhead) + Σ(memory_transfer) + idle_time

很多时候：
- launch_overhead > kernel_time（小 kernel 问题）
- idle_time > compute_time（流水线断裂）
- memory_transfer 可以与 compute 重叠（但没做）
```

### Level 3: 系统层（顶尖工程师的思维）

```
问题：这个算法/架构在这个硬件上的理论极限是什么？我们距离极限还有多远？
工具：Roofline 模型、数学分析、硬件手册
优化：算法选择、数据流重组、架构调整
提升：10x - 100x
```

**核心问题**：
```python
# 错误的问题
"如何让这个 GEMM kernel 更快？"

# 正确的问题
"在 A100 上，FP16 GEMM 的理论峰值是 312 TFLOPS
 我们的实现达到了 180 TFLOPS (58%)
 剩余的 42% 在哪里？能否突破 90%？"
```

---

## Roofline 模型：性能的物理极限

### 什么是 Roofline？

Roofline 模型告诉你：**在特定硬件上，给定 Arithmetic Intensity，性能的物理上限是多少**

```
              Performance (TFLOPS)
                    │
Peak Compute (312)  │         ┌─────────────── Compute Roof
                    │        ╱
                    │       ╱
                    │      ╱
                    │     ╱  Diagonal = Memory Roof
                    │    ╱   slope = Bandwidth
                    │   ╱
                    │  ╱
                    │ ╱
                    │╱
────────────────────┼──────────────────────────► AI (FLOPS/Byte)
                    0    1    10   100  1000
                         │
                    Ridge Point = Peak Compute / Peak Bandwidth
```

### A100 的 Roofline 参数

| 硬件参数 | FP32 | FP16 (Tensor Core) |
|---------|------|-------------------|
| Peak Compute | 19.5 TFLOPS | 312 TFLOPS |
| Peak Bandwidth | 1.5 TB/s | 1.5 TB/s |
| Ridge Point | 13 FLOPS/Byte | 208 FLOPS/Byte |

**关键洞察**：

```python
AI = FLOPS / Bytes_Accessed

# Matrix Multiplication (M=N=K=4096, FP16)
FLOPS = 2 * M * N * K = 137 GFLOPS
Bytes = 2 * (M*K + K*N + M*N) * 2 bytes = 201 MB
AI = 137 / 0.2 = 685 FLOPS/Byte

685 > 208 → Compute-bound → 理论峰值 = 312 TFLOPS ✓

# Element-wise Add (N=1M, FP32)
FLOPS = N = 1M
Bytes = 3 * N * 4 bytes = 12 MB
AI = 1 / 12 = 0.083 FLOPS/Byte

0.083 << 13 → Memory-bound → 理论峰值 = 0.083 * 1.5TB/s = 125 GFLOPS
```

### 使用 Roofline 指导优化

#### 案例 1：LayerNorm (典型的 Memory-bound)

```python
# LayerNorm: mean, var, normalize
# Input: [B, N] = [128, 4096]
# FLOPS: ~4 * B * N = 2M FLOPS
# Bytes: ~4 * B * N * 4 bytes = 8 MB (read) + 8 MB (write) = 16 MB
# AI = 2M / 16M = 0.125 FLOPS/Byte

理论峰值 = 0.125 * 1.5 TB/s = 187.5 GFLOPS
实际测量 = 145 GFLOPS
效率 = 145 / 187.5 = 77% ← 已经很好了！

结论：不要在这里浪费时间优化 kernel 细节，77% 已经接近极限
```

**关键决策**：如果一个 Memory-bound kernel 已经达到带宽的 75%+，继续优化 kernel 内部是浪费时间，应该：
1. 融合相邻 kernel 减少内存访问
2. 或者接受这个性能，优化其他地方

#### 案例 2：GEMM (Compute-bound)

```python
# GEMM: C = A @ B
# A: [4096, 4096], B: [4096, 4096], FP16
# FLOPS: 2 * 4096^3 = 137 GFLOPS
# Bytes (no reuse): (4096^2 + 4096^2 + 4096^2) * 2 = 100 MB
# Bytes (optimal with blocking): 理论最小 ~ 3 * 4096 * sqrt(cache_size)

AI > 208 → Compute-bound

理论峰值 (Tensor Core) = 312 TFLOPS
cuBLAS 实际 = 280 TFLOPS (90% 效率) ← 已经极致优化
朴素实现 = 15 TFLOPS (5% 效率) ← 别自己写 GEMM！

结论：使用 cuBLAS / cuDNN，不要自己写
```

### Roofline 分析工具

```bash
# NCU 自动生成 Roofline 分析
ncu --set roofline -o profile ./program

# 在 NCU GUI 中查看：
# Details -> Roofline Analysis
# 会显示你的 kernel 在 Roofline 图上的位置
```

---

## Amdahl's Law：优化的投资回报率

### Amdahl's Law 基础

```
整体加速比 = 1 / ((1 - P) + P / S)

P = 被优化部分占总时间的比例
S = 该部分的加速比
```

### 实战应用：优化决策树

假设你有一个端到端时间为 100ms 的程序：

```
Profile 结果：
- Kernel A: 50ms (50%)
- Kernel B: 30ms (30%)
- Kernel C: 10ms (10%)
- Memory Copy: 5ms (5%)
- Others: 5ms (5%)
```

**决策矩阵**：

| 优化目标 | 预期加速 | 优化时间 | 整体提升 | ROI |
|---------|---------|---------|---------|-----|
| Kernel A (2x) | 50ms → 25ms | 2 天 | 100ms → 75ms (1.33x) | ⭐⭐⭐⭐⭐ |
| Kernel B (3x) | 30ms → 10ms | 3 天 | 100ms → 80ms (1.25x) | ⭐⭐⭐ |
| Kernel C (10x) | 10ms → 1ms | 5 天 | 100ms → 91ms (1.1x) | ⭐ |
| A+B 融合 (1.5x) | 80ms → 53ms | 1 天 | 100ms → 73ms (1.37x) | ⭐⭐⭐⭐⭐ |

**量化公式**：

```python
def calculate_roi(original_time, part_ratio, speedup, dev_days):
    """计算优化的 ROI"""
    new_time = original_time * (1 - part_ratio + part_ratio / speedup)
    overall_speedup = original_time / new_time

    # 假设每天开发成本 = 1
    roi = (overall_speedup - 1) / dev_days

    return overall_speedup, roi

# 例子
print(calculate_roi(100, 0.5, 2, 2))  # (1.33x, 0.165)
print(calculate_roi(100, 0.1, 10, 5)) # (1.1x, 0.02)
```

**结论**：优先优化占时间最多的部分，即使加速比不是最高

### 现实中的 Amdahl's Law

```python
# 错误思维
"我把这个 kernel 从 1ms 优化到 0.1ms，加速 10x！"
→ 但这个 kernel 只占总时间的 2%
→ 总体只快了 1.8%
→ 浪费了 3 天

# 正确思维
"这个 kernel 占总时间 40%，即使只优化 50%，总体也能快 20%"
→ 优先优化这个
→ ROI 高 10 倍
```

---

## 系统级瓶颈分析框架

### 五维分析模型

```
端到端性能 = f(Compute, Memory, Launch, Transfer, Sync)
```

#### 1. Compute（计算瓶颈）

**识别**：
```bash
nsys profile ./program
# 查看 GPU Utilization
# 如果 SM Active > 80%，大部分时间都在计算 → Compute-bound
```

**量化**：
```python
实际 FLOPS / 理论峰值 FLOPS = 计算效率

# A100 FP16 Tensor Core
理论峰值 = 312 TFLOPS

# 如果测到 280 TFLOPS → 90% 效率 → 已接近极限
# 如果测到 150 TFLOPS → 48% 效率 → 还有优化空间
```

**优化方向**：
- Kernel 层：提高 occupancy、减少 divergence、使用 Tensor Cores
- 算法层：选择计算复杂度更低的算法（如 Flash Attention）

#### 2. Memory（内存瓶颈）

**识别**：
```bash
ncu --section SpeedOfLight ./program
# Memory Throughput > 80% → Memory-bound
```

**量化**：
```python
实际带宽 / 理论峰值带宽 = 带宽效率

# A100
理论峰值 HBM 带宽 = 1.5 TB/s

# 如果测到 1.2 TB/s → 80% 效率 → 接近极限
# 如果测到 600 GB/s → 40% 效率 → 大量优化空间
```

**深入分析**：Memory 瓶颈有多种类型：

```
Memory 层次结构：
HBM (1.5 TB/s, 40GB)
  ↕ ~200 cycles
L2 Cache (6 MB, ~5 TB/s)
  ↕ ~50 cycles
L1/Shared (192 KB/SM, ~20 TB/s)
  ↕ ~5 cycles
Registers (64K regs/SM, ~100 TB/s)
  ↕ ~1 cycle
```

**案例分析**：

```python
# 案例：Element-wise 操作慢

# 第一步：检查是哪一层的内存瓶颈
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    lts__t_sectors_srcunit_tex_op_read.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    ./program

输出：
dram__throughput = 85% ← HBM 接近饱和
lts (L2) hit rate = 20% ← L2 几乎没缓存
l1tex hit rate = 95% ← L1 缓存很好

诊断：数据没有跨 kernel 复用，每次都从 HBM 读取
优化：算子融合，让数据留在 L1/L2

# 第二步：检查访问模式
ncu --metrics l1tex__average_t_sectors_per_request ./program

输出：
l1tex__average_t_sectors_per_request = 3.2

理想值 = 1.0 (完美 coalesced)
实际值 = 3.2 → 每次请求浪费了 3.2 - 1 = 2.2 个 sector
浪费带宽 = 68%

优化：修复访问模式（见 03_performance_optimization）
```

#### 3. Launch Overhead（启动开销）

**识别**：
```bash
nsys profile ./program
# 查看 Timeline
# 如果看到大量 gaps between kernels → Launch overhead 问题
```

**量化**：
```python
# Kernel launch overhead ≈ 5-10 μs per kernel

# 案例：100 个小 kernel，每个 50 μs
kernel_time = 100 * 50 = 5000 μs = 5 ms
launch_overhead = 100 * 7 = 700 μs = 0.7 ms
overhead_ratio = 0.7 / 5 = 14%

# 融合后：1 个大 kernel，100 * 50 = 5000 μs
kernel_time = 5 ms
launch_overhead = 7 μs
overhead_ratio = 0.14%

加速比 = 5.7 / 5.007 = 1.14x
```

**优化策略**：
```python
# 1. 算子融合（最有效）
加速比：1.1x - 10x (取决于 kernel 数量和大小)

# 2. CUDA Graphs（减少 CPU overhead）
加速比：1.05x - 1.3x (对小 kernel 密集的 workload)

# 3. Persistent Kernels（高级）
概念：让 kernel 一直运行，通过 shared memory 或 global memory 通信
加速比：1.2x - 2x
适用：高频调用的小 kernel
```

#### 4. Transfer（数据传输）

**识别**：
```bash
nsys profile --trace=cuda,nvtx ./program
# 查看 Memory Operations
# 如果 H2D/D2H copy 时间 > 10% → Transfer 瓶颈
```

**量化**：
```python
# PCIe 3.0 x16: ~12 GB/s (双向各 12 GB/s)
# PCIe 4.0 x16: ~24 GB/s

# 案例：传输 1GB 数据
理论时间 (PCIe 3.0) = 1GB / 12GB/s = 83 ms
如果测量到 90 ms → 效率 = 92% → 正常
如果测量到 200 ms → 效率 = 41% → 有问题

可能原因：
- 使用了 pageable memory（不是 pinned）
- 小数据多次传输（overhead 大）
- 没有使用异步传输
```

**优化策略**：

```python
# 1. 减少传输（最优先！）
# 尽量在 GPU 上完成所有操作
加速比：无限大 (0ms vs 100ms)

# 2. Pinned Memory
import torch

# ❌ Pageable memory
data = torch.randn(1000000)
data_gpu = data.cuda()  # 慢！~200 MB/s

# ✅ Pinned memory
data = torch.randn(1000000).pin_memory()
data_gpu = data.cuda()  # 快！~12 GB/s

加速比：10x - 100x

# 3. 异步传输 + 重叠
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    data1_gpu = data1.cuda(non_blocking=True)
    kernel1(data1_gpu)

with torch.cuda.stream(stream2):
    data2_gpu = data2.cuda(non_blocking=True)  # 与 kernel1 重叠
    kernel2(data2_gpu)

加速比：1.5x - 2x
```

#### 5. Sync（同步开销）

**识别**：
```python
# 代码中每次调用这些都会同步：
torch.cuda.synchronize()
tensor.cpu()  # 隐式同步
tensor.item()  # 隐式同步
print(tensor)  # 隐式同步（如果是 GPU tensor）

# 在 nsys 中看到大量 cudaDeviceSynchronize → Sync 问题
```

**案例**：

```python
# ❌ 错误：频繁同步
for i in range(1000):
    loss = model(data)
    print(f"Loss: {loss.item()}")  # 每次都同步！← 极慢

# 测量：1000 次迭代用了 50 秒
# 其中 sync overhead = 1000 * 10μs = 10ms... 不对，为什么这么慢？

# 真相：每次 .item() 都会：
# 1. GPU kernel 执行（异步）
# 2. .item() 触发同步，等待 GPU 完成
# 3. D2H 传输
# 4. CPU 处理
# → 完全破坏了 GPU 的并行流水线！

# ✅ 正确：批量同步
losses = []
for i in range(1000):
    loss = model(data)
    losses.append(loss)

torch.cuda.synchronize()  # 只同步一次
losses_cpu = [l.item() for l in losses]

# 测量：1000 次迭代用了 5 秒
加速比：10x
```

---

## 实战案例：Transformer 训练优化

### 场景

```
模型：GPT-2 (124M parameters)
硬件：A100 40GB
任务：优化训练吞吐量 (samples/sec)
```

### Baseline Profile

```bash
nsys profile -o baseline python train.py --profile
```

**结果**：

```
端到端：每个 batch 500 ms

分解：
- Forward pass: 200 ms (40%)
  - Self-Attention: 120 ms (24%)
  - FFN: 60 ms (12%)
  - LayerNorm: 20 ms (4%)
- Backward pass: 250 ms (50%)
- Optimizer step: 30 ms (6%)
- Data loading: 15 ms (3%)
- Others: 5 ms (1%)
```

### 优化路径

#### 第一步：Attention 优化 (最大瓶颈)

**问题诊断**：

```python
# 标准 Attention 实现
Q = x @ Wq  # [B, N, D]
K = x @ Wk
V = x @ Wv

scores = Q @ K.T  # [B, N, N] ← 大矩阵！
scores = scores / sqrt(D)
attn = softmax(scores)
output = attn @ V

# 问题分析：
# 1. 中间矩阵 scores: [B, N, N] = [32, 2048, 2048] = 128 MB
# 2. Memory 操作：
#    - QK^T: 读 Q (16MB) + K (16MB), 写 scores (128MB) = 160 MB
#    - Softmax: 读写 scores = 256 MB
#    - Attn @ V: 读 attn (128MB) + V (16MB), 写 output (16MB) = 160 MB
#    总共：~576 MB

# 3. 计算量：
#    FLOPS = 2*B*N*N*D + 2*B*N*N*D = 4*B*N*N*D = 4*32*2048*2048*64 = 34 GFLOPS
#    AI = 34 GFLOPS / 576 MB = 0.06 FLOPS/Byte
#
#    → Memory-bound（AI << 13）
#    → 理论峰值 = 0.06 * 1.5 TB/s = 90 GFLOPS
#    → 实际测量 = 70 GFLOPS → 78% 效率

# 结论：标准 Attention 的瓶颈是中间矩阵的内存访问
```

**优化：Flash Attention**

```python
# Flash Attention 核心思想：
# 1. 分块计算，避免实体化整个 attention matrix
# 2. 使用 SRAM (shared memory) 而不是 HBM

# 理论分析：
# Memory 访问（Flash Attention）：
#   - 优化后 ~50 MB (相比 576 MB)
#   - IO 复杂度：O(N^2 d^2 / M)，其中 M = SRAM size
#
# 时间：120 ms → 40 ms (3x 加速)
# 总体：500 ms → 380 ms (1.32x)
```

**关键洞察**：
```
有时算法级别的优化比 kernel 级别优化重要 100 倍

标准 Attention kernel 优化到极致：78% → 95% 效率
→ 提升：120 ms → 93 ms (1.3x)

换成 Flash Attention:
→ 提升：120 ms → 40 ms (3x)

ROI 差距：Flash Attention 的 ROI 是 kernel 优化的 10 倍
```

#### 第二步：混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Forward 使用 FP16
with autocast():
    output = model(input)
    loss = criterion(output, target)

# Backward
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 结果：
# - Forward: 200 ms → 120 ms (1.67x)
# - Backward: 250 ms → 150 ms (1.67x)
# - 总体：380 ms → 250 ms (1.52x)

# 为什么这么快？
# 1. FP16 GEMM 使用 Tensor Cores: 19.5 → 312 TFLOPS (16x 峰值算力)
# 2. FP16 内存带宽需求减半: 2 bytes vs 4 bytes
# 3. 实际测量：Tensor Core 利用率从 0% → 85%
```

#### 第三步：Gradient Checkpointing（内存换时间的反例）

```python
# 问题：大模型 OOM

# Gradient Checkpointing 原理：
# - Forward: 不保存中间激活值（除了 checkpoints）
# - Backward: 重新计算中间激活值

# 权衡：
# 内存：减少 ~50% (40GB → 20GB)
# 时间：增加 ~30% (250ms → 325ms)

# 但总体：
# 之前：batch_size = 16, time = 250ms
# 现在：batch_size = 32, time = 325ms
# 吞吐量：(16/250) → (32/325) = 0.064 → 0.098 samples/ms (1.53x)

# 结论：虽然单个 batch 变慢了，但能跑更大的 batch，总吞吐量提升
```

#### 第四步：Operator Fusion

```python
# Profile 发现：LayerNorm 很频繁
nsys profile --trace=cuda ./train.py

# 输出：
# - layernorm_kernel: 被调用 100 次，每次 200μs，总 20ms
# - 但 kernel 之间有 gap

# 优化：Fused LayerNorm + Linear
# PyTorch 2.0: torch.compile 自动融合

@torch.compile
def fused_block(x, ln_weight, ln_bias, linear_weight, linear_bias):
    x = F.layer_norm(x, normalized_shape, ln_weight, ln_bias)
    x = F.linear(x, linear_weight, linear_bias)
    return x

# 结果：
# - LayerNorm + Linear: 20ms + 15ms = 35ms
# - Fused: 25ms
# - 加速：1.4x (局部), 1.04x (全局)
```

### 最终结果

```
优化路径汇总：

Baseline: 500 ms/batch, 32 samples/sec

1. Flash Attention:  500 → 380 ms (1.32x) → 42 samples/sec
2. Mixed Precision:  380 → 250 ms (1.52x) → 64 samples/sec
3. Larger Batch:     250 → 325 ms (but batch 16→32) → 98 samples/sec
4. Operator Fusion:  325 → 312 ms (1.04x) → 102 samples/sec

总体加速：3.2x (32 → 102 samples/sec)
```

**关键教训**：
1. **算法优化 >> kernel 优化** (Flash Attention 带来最大提升)
2. **硬件特性很重要** (Tensor Cores)
3. **有时变慢是为了更快** (Gradient Checkpointing)
4. **Amdahl's Law** (优先优化 Attention，而不是 LayerNorm)

---

## 反直觉的优化策略

### 1. "慢"的算法可能更快

**案例：Sorting**

```python
# GPU 上：

# QuickSort:
# - CPU: O(n log n), 非常快
# - GPU: 大量分支，warp divergence 严重
# 实际性能：100ms for 10M elements

# Radix Sort:
# - CPU: O(nk), 通常比 QuickSort 慢
# - GPU: 无分支，完美并行
# 实际性能：10ms for 10M elements

# 结论：在 GPU 上，O(nk) 的算法比 O(n log n) 快 10 倍
```

**原因**：
```
理论复杂度忽略了常数因子和硬件特性

GPU 性能 = f(算法复杂度, 并行度, 内存访问, 分支)

有时：
  高复杂度 + 高并行度 + 无分支 > 低复杂度 + 低并行度 + 多分支
```

### 2. 更多的计算可能减少时间

**案例：Im2Col**

```python
# 卷积的两种实现：

# 方法 1：直接卷积
# - 计算量：O(C_in * C_out * K * K * H * W)
# - 内存访问：复杂，不规则
# - 实际性能：150 ms

# 方法 2：Im2Col + GEMM
# - 展开 input: [B, C, H, W] → [B*H*W, C*K*K]
# - 计算量：更多（展开有冗余）！
# - 内存访问：GEMM 极度优化（cuBLAS）
# - 实际性能：50 ms

# 结论：冗余计算 + 极致优化的 GEMM > 精确计算 + 复杂访问
```

**原因**：
```
cuBLAS GEMM 的优化程度：~90% 峰值性能
手写卷积 kernel 的优化程度：~30% 峰值性能

即使 GEMM 做了 2 倍的计算，仍然更快：
  0.9 * 2 > 0.3 * 1 → 还快了 6 倍
```

### 3. 占用率不是越高越好

**反直觉案例**：

```python
# Kernel A:
# - Occupancy: 100%
# - Shared Memory: 0
# - Performance: 100 ms

# Kernel B:
# - Occupancy: 50% (因为用了大量 Shared Memory)
# - Shared Memory: 96 KB/SM
# - Performance: 60 ms

# 为什么 Occupancy 低反而更快？

# 答案：Occupancy 只是"可能并行的 warps 数量"
#       但如果这些 warps 都在等内存，多了也没用

# Kernel B 用 Shared Memory 缓存了数据：
# - 减少了 HBM 访问
# - 虽然 Occupancy 低，但每个 warp 真正在计算
# - 结果：更快
```

**量化分析**：

```
Kernel A:
  - 32 个 warps 同时运行
  - 每个 warp 90% 时间在等 HBM (stalled)
  - 有效计算时间：32 * 10% = 3.2 warps

Kernel B:
  - 16 个 warps 同时运行
  - 每个 warp 30% 时间在等 Shared Memory (快得多)
  - 有效计算时间：16 * 70% = 11.2 warps

11.2 > 3.2 → Kernel B 快 3.5 倍
```

**结论**：关注"有效占用率" (Active Warps)，而不是"理论占用率"

### 4. 融合不一定总是好的

**案例**：

```python
# 两个 kernel：
# Kernel A: GEMM (4096x4096), compute-bound, 80% Tensor Core 利用率
# Kernel B: Element-wise Add, memory-bound

# 融合后：
# - 无法使用 Tensor Cores（mixed operation）
# - Compute-bound 变成 memory-bound
# - 性能：30 ms + 5 ms = 35 ms → 融合后 50 ms（更慢！）

# 结论：不要融合 compute-bound 和 memory-bound kernel
```

**什么时候融合**：
```
✅ 融合：
  - 多个 memory-bound element-wise kernels
  - 中间结果只用一次
  - 融合后仍在同一个 bound 类型

❌ 不融合：
  - 一个 compute-bound + 一个 memory-bound
  - 中间结果被多次使用
  - 融合后导致寄存器溢出
```

### 5. 最快的代码是不写的代码

**案例：Dropout**

```python
# Training: 需要 Dropout
# Inference: 不需要 Dropout

# ❌ 很多代码在 inference 时仍然调用 dropout（虽然 p=0）
if self.training:
    x = F.dropout(x, p=0.1)
else:
    x = F.dropout(x, p=0.0)  # ← 仍然调用了 kernel！

# ✅ 正确做法
if self.training:
    x = F.dropout(x, p=0.1)
# else: 什么都不做

# 性能差异：10 ms vs 0 ms
```

**教训**：
- 删除代码是最好的优化
- 检查 inference 代码路径，删除所有训练相关的操作
- 使用 `torch.jit.script` 自动做这些优化

---

## Production 环境的隐藏陷阱

### 陷阱 1：Batch Size 动态变化

```python
# Lab 环境：固定 batch_size = 32, 性能很好

# Production:
# - 真实请求：batch 大小不固定 (1, 5, 16, 32, 64...)
# - CUDA kernel 对不同 batch size 性能差异巨大

# 测量：
batch_size=1:  50 ms/sample
batch_size=32: 5 ms/sample  (10x faster per sample!)
batch_size=64: 3 ms/sample

# 问题：为什么差这么多？

# 原因 1：Launch overhead 被摊薄
launch_overhead = 10 μs
batch=1:  10 μs / 1 = 10 μs per sample
batch=32: 10 μs / 32 = 0.3 μs per sample

# 原因 2：硬件利用率
batch=1:  80 个 SM, 只用了 10 个 (12%)
batch=32: 80 个 SM, 用了 75 个 (94%)

# 解决方案：Dynamic Batching
# 聚合多个小请求到一个 batch
```

**Dynamic Batching 实现**：

```python
import asyncio
import torch

class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []

    async def infer(self, x):
        """单个请求"""
        future = asyncio.Future()
        self.queue.append((x, future))

        # 如果达到 max_batch_size，立即执行
        if len(self.queue) >= self.max_batch_size:
            await self._execute_batch()

        # 否则等待 max_wait_ms
        try:
            await asyncio.wait_for(future, timeout=self.max_wait_ms/1000)
        except asyncio.TimeoutError:
            await self._execute_batch()

        return await future

    async def _execute_batch(self):
        if not self.queue:
            return

        batch_x = torch.stack([x for x, _ in self.queue])
        batch_y = self.model(batch_x)

        for (_, future), y in zip(self.queue, batch_y):
            future.set_result(y)

        self.queue.clear()

# 使用
batcher = DynamicBatcher(model)

# 多个并发请求自动 batching
results = await asyncio.gather(*[
    batcher.infer(x1),
    batcher.infer(x2),
    # ...
])
```

### 陷阱 2：显存碎片化

```python
# 现象：训练跑了几个小时后 OOM，但一开始没问题

# 原因：显存碎片化

# PyTorch 内存分配器：
# - 分配时：找一个足够大的 block
# - 释放时：标记为 free，但不归还 CUDA
# - 问题：长时间运行后，大量小碎片

# 诊断：
print(torch.cuda.memory_allocated())  # 实际使用
print(torch.cuda.memory_reserved())   # 从 CUDA 申请的

# 如果 reserved >> allocated → 碎片化严重

# 解决方案 1：定期清空缓存
if iteration % 1000 == 0:
    torch.cuda.empty_cache()

# 解决方案 2：使用 memory pool 策略
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 解决方案 3：固定内存分配（最彻底）
# 预分配所有需要的 tensor，训练时复用
```

### 陷阱 3：多 GPU 的隐藏同步

```python
# DataParallel (DP) vs DistributedDataParallel (DDP)

# ❌ DataParallel (慢)
model = nn.DataParallel(model)

# 问题：
# 1. 主 GPU (GPU 0) 负载不均
# 2. 每次 forward/backward 都要同步
# 3. GIL 限制

# 测量（4x A100）：
# - 理论：4x 加速
# - DP 实际：2.5x
# - 效率：62.5%

# ✅ DistributedDataParallel (快)
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])

# 测量（4x A100）：
# - DDP 实际：3.8x
# - 效率：95%

# 为什么 DDP 快？
# - 每个 GPU 独立运行
# - 只在 backward 结束时同步 gradients
# - 使用 NCCL all-reduce (高度优化)
```

### 陷阱 4：CPU 瓶颈

```python
# 现象：GPU 利用率只有 60%

# nsys profile 发现：GPU 在等 CPU

# 常见原因：

# 1. 数据预处理太慢
# ❌
for batch in dataloader:  # CPU preprocessing
    batch_gpu = batch.cuda()
    model(batch_gpu)

# ✅
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,        # 多进程
    pin_memory=True,      # 避免 pageable memory
    prefetch_factor=4,    # 预取
)

# 2. Metric 计算太频繁
# ❌
for batch in dataloader:
    output = model(batch)
    acc = calculate_accuracy(output, target)  # 每个 batch 都计算

# ✅
for batch in dataloader:
    output = model(batch)
    outputs.append(output)  # 累积

acc = calculate_accuracy(torch.cat(outputs), targets)  # 最后计算一次

# 3. 同步太频繁
# ❌
for i in range(1000):
    loss = model(data)
    if loss.item() > threshold:  # .item() 同步！
        break

# ✅
for i in range(1000):
    loss = model(data)
    losses.append(loss)

torch.cuda.synchronize()
if any(l.item() > threshold for l in losses):
    break
```

### 陷阱 5：CUDA Kernel 版本不匹配

```python
# 问题：在 A100 上优化的代码，在 V100 上更慢

# 原因：架构不同

# A100 (Ampere, sm_80):
# - Tensor Cores for FP32
# - Async copy
# - 更大的 L2 cache (40 MB)

# V100 (Volta, sm_70):
# - 没有 FP32 Tensor Cores
# - 同步 copy
# - 小 L2 (6 MB)

# 解决方案：Runtime detection

import torch

def get_gemm_kernel(device):
    capability = torch.cuda.get_device_capability(device)

    if capability >= (8, 0):  # Ampere+
        return gemm_tensor_core_fp32
    elif capability >= (7, 0):  # Volta+
        return gemm_tensor_core_fp16
    else:
        return gemm_cuda_core

# 或使用 Triton auto-tuning
# Triton 会在目标硬件上自动选择最优配置
```

---

## 总结：专家级优化的思维模型

### 1. 建立性能直觉

```python
# 初级工程师：盲目优化
"我改了这里，跑一下看看快不快"

# 高级工程师：量化预测
"理论峰值是 X，现在是 Y，差距是 Z，问题在 W"

# 专家：建立 mental model
"根据 Roofline，这个 kernel 的 AI 是 0.5，所以是 memory-bound
 根据 Amdahl's Law，优化这个只能提升 15%
 根据硬件手册，L2 带宽是 5 TB/s，我们只用了 60%
 → 应该优化 L2 cache 复用，而不是 L1 或寄存器"
```

### 2. 优化的优先级

```
第一优先级（10x+ 提升）：
├─ 算法选择 (Flash Attention, Fused kernels)
├─ 硬件特性 (Tensor Cores, Mixed Precision)
└─ 系统架构 (Multi-GPU, Pipeline Parallelism)

第二优先级（2-5x 提升）：
├─ Operator Fusion
├─ Memory Access Patterns
└─ Occupancy Optimization

第三优先级（1.1-1.5x 提升）：
├─ ILP, Loop Unrolling
├─ CUDA Graphs
└─ Kernel细节优化
```

### 3. 思考清单

每次优化前问自己：

```
□ 这个优化能提升多少？(量化)
□ 需要多少时间？(工程成本)
□ ROI 是多少？(收益/成本)
□ 有没有更高 ROI 的优化？(opportunity cost)
□ 优化后的理论极限是多少？(天花板在哪)
□ 我们离理论极限还有多远？(优化空间)
```

### 4. 工具链

```
Stage 1: 全局分析 (10 分钟)
└─ nsys profile → 找到热点

Stage 2: 瓶颈定位 (30 分钟)
├─ Roofline 分析 → 理论极限
├─ Amdahl's Law → 优化优先级
└─ 五维分析 → 具体瓶颈类型

Stage 3: 深入优化 (几小时到几天)
├─ 算法级：Flash Attention, 更好的算法
├─ 系统级：Fusion, Multi-stream, CUDA Graphs
└─ Kernel级：NCU 深入分析，内存优化

Stage 4: 验证 (1 小时)
├─ 性能测试：端到端时间
├─ Profile 对比：优化前后差异
└─ 正确性验证：数值精度
```

---

## 参考资料

### 论文
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Roofline: An Insightful Visual Performance Model](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)

### 工具文档
- [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 深入阅读
- GPU 架构：[NVIDIA Ampere Architecture](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- 性能模型：[Performance Analysis and Tuning on Modern CPUs and GPUs](http://cs.utexas.edu/~pingali/CS395T/2021sp/)
