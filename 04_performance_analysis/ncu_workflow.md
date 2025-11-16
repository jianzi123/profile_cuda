# NCU 性能分析完整工作流

## 目录
1. [NCU 使用快速入门](#ncu-使用快速入门)
2. [第一步：Speed of Light 分析](#第一步speed-of-light-分析)
3. [第二步：瓶颈深入分析](#第二步瓶颈深入分析)
4. [第三步：优化机会识别](#第三步优化机会识别)
5. [完整分析检查清单](#完整分析检查清单)
6. [实战案例](#实战案例)

---

## NCU 使用快速入门

### 基础命令

```bash
# 1. 快速分析（推荐起点）
ncu --set full -o profile_result ./your_program

# 2. 只分析特定 kernel
ncu --set full -o profile_result \
    --kernel-name your_kernel_name \
    ./your_program

# 3. 分析前 N 次调用
ncu --set full -o profile_result \
    --kernel-name your_kernel_name \
    --launch-count 10 \
    ./your_program

# 4. 导出为文本格式（命令行查看）
ncu --set full \
    --page raw \
    --csv \
    ./your_program > results.csv

# 5. 打开 GUI 查看
ncu-ui profile_result.ncu-rep
```

### 推荐的 Section 组合

```bash
# 最小集（快速诊断）
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    ./your_program

# 标准集（日常使用）
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    ./your_program

# 完整集（深度分析）
ncu --set full ./your_program
```

---

## 第一步：Speed of Light 分析

### 这一步的目标

**快速判断算子的性能瓶颈类型**

### 命令

```bash
ncu --section SpeedOfLight -o step1 ./program
ncu-ui step1.ncu-rep
```

### 关键指标

在 NCU GUI 中，打开 "Details" → "Speed of Light Throughput"：

#### 1. SM Throughput (计算吞吐量)

```
指标名：sm__throughput.avg.pct_of_peak_sustained_elapsed
单位：% of peak

含义：计算单元的利用率
```

**如何解读**：
```
> 80%  →  ✅ Compute-bound (计算受限)
           → 优化方向：算法优化、减少计算量

60-80% →  ⚠️  Balanced (平衡状态)
           → 既有计算也有内存瓶颈

40-60% →  ⚠️  Moderate utilization
           → 可能有其他问题（occupancy、divergence）

< 40%  →  ❌ Under-utilized (未充分利用)
           → 可能是 Memory-bound 或其他问题
```

#### 2. Memory Throughput (内存吞吐量)

```
指标名：gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
单位：% of peak

含义：整体内存系统的利用率
```

**如何解读**：
```
> 80%  →  ✅ Memory-bound (内存受限)
           → 优化方向：减少内存访问、提高缓存命中率

60-80% →  ⚠️  High memory pressure
           → 内存是主要瓶颈之一

< 60%  →  ✓  Memory not a bottleneck
           → 瓶颈在其他地方
```

#### 3. 组合判断

| SM % | Memory % | 结论 | 优化方向 |
|------|----------|------|---------|
| > 80 | < 60 | **Compute-bound** | 算法优化、减少 FLOPs |
| < 60 | > 80 | **Memory-bound** | 内存优化、缓存优化 |
| > 70 | > 70 | **Well-balanced** | 两方面都接近极限 |
| < 50 | < 50 | **Under-utilized** | 查看 Occupancy、Warp State |

### 实际示例

#### 示例 1：典型的 Memory-bound Kernel

```
=== Speed of Light ===
SM Throughput:              15%     ← 计算单元几乎空闲
Memory Throughput:          92%     ← 内存带宽接近饱和

DRAM Throughput:            85%
L2 Cache Throughput:        78%
L1/TEX Cache Throughput:    65%

结论：Memory-bound！
优化方向：
1. 减少全局内存访问
2. 利用 Shared Memory
3. 提高 L1/L2 缓存命中率
4. 向量化访问（float4）
```

#### 示例 2：典型的 Compute-bound Kernel

```
=== Speed of Light ===
SM Throughput:              88%     ← 计算单元高度利用
Memory Throughput:          45%     ← 内存不是瓶颈

DRAM Throughput:            35%
FP32 Throughput:            85%

结论：Compute-bound！
优化方向：
1. 使用 Tensor Cores（如果适用）
2. 使用快速数学函数
3. 减少不必要的计算
4. 提高 ILP
```

#### 示例 3：低利用率（需要进一步分析）

```
=== Speed of Light ===
SM Throughput:              25%     ← 低利用率
Memory Throughput:          30%     ← 低利用率

结论：既不是 compute-bound 也不是 memory-bound
可能问题：
1. Low Occupancy（占用率低）
2. Warp Divergence（分支发散）
3. Excessive Synchronization（过度同步）
4. Launch Configuration 问题

→ 需要进入第二步分析
```

---

## 第二步：瓶颈深入分析

根据第一步的结论，选择对应的分析路径。

### 路径 A：Memory-bound 深入分析

#### 命令

```bash
ncu --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    -o memory_analysis ./program
```

#### A1. 查看内存层次利用率

**关键指标**：

##### DRAM (Global Memory)

```
指标：dram__throughput.avg.pct_of_peak_sustained_elapsed
单位：%

> 80%: DRAM 是瓶颈
```

**详细指标**：
```
dram__bytes_read.sum                    # DRAM 读取字节数
dram__bytes_write.sum                   # DRAM 写入字节数
dram__sectors_read.sum                  # 读取扇区数
dram__sectors_write.sum                 # 写入扇区数
```

**计算实际带宽**：
```python
total_bytes = dram__bytes_read + dram__bytes_write
duration_sec = gpu__time_duration.sum / 1e9
actual_bandwidth = total_bytes / duration_sec  # bytes/sec
theoretical_bandwidth = 1555e9  # A100 = 1555 GB/s

bandwidth_utilization = actual_bandwidth / theoretical_bandwidth * 100
```

##### L2 Cache

```
指标：lts__t_sectors.avg.pct_of_peak_sustained_elapsed
单位：%

> 80%: L2 cache 是瓶颈
```

**命中率**：
```
指标：lts__t_sector_hit_rate.pct
单位：%

理想值：> 70%
< 50%: L2 利用不充分，数据没有复用
```

**详细指标**：
```
lts__t_sectors_op_read.sum              # L2 读扇区
lts__t_sectors_op_write.sum             # L2 写扇区
lts__t_sector_hit_rate.pct              # L2 命中率
```

##### L1/TEX Cache

```
指标：l1tex__throughput.avg.pct_of_peak_sustained_elapsed
单位：%
```

**命中率**：
```
指标：l1tex__t_sector_hit_rate.pct
单位：%

理想值：> 90%
< 70%: L1 利用不充分
```

**详细指标**：
```
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   # L1 全局加载
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum   # L1 全局存储
l1tex__t_sector_hit_rate.pct                      # L1 命中率
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
    # 每次请求的平均扇区数
    # 理想值 = 1 (完美 coalesced)
    # > 1: 非 coalesced 访问
```

#### A2. 判断 Coalesced Access（合并访问）

**关键指标**：

```
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
```

**如何解读**：
```
= 1.0  →  ✅ 完美 coalesced (每个请求正好 1 个扇区)
= 2.0  →  ⚠️  浪费 50% 带宽
= 4.0  →  ❌ 浪费 75% 带宽（跨步访问或随机访问）
```

**示例分析**：
```
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld = 4.2

分析：
- 每次加载请求需要 4.2 个扇区
- 理想情况是 1 个扇区
- 浪费：(4.2 - 1) / 4.2 = 76% 的带宽！

可能原因：
1. 跨步访问（strided access）
2. 随机访问（random access）
3. 未对齐的访问

优化方向：
1. 修改访问模式为连续访问
2. 使用 Shared Memory
3. 向量化访问（float4）
```

#### A3. 检查 Shared Memory Bank Conflicts

**关键指标**：

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  # 加载冲突
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  # 存储冲突
```

**如何解读**：
```
= 0     →  ✅ 无 bank conflicts
> 0     →  ⚠️  有 bank conflicts

每次冲突都会串行化访问，降低性能
```

**计算冲突率**：
```python
shared_accesses = l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
bank_conflicts = l1tex__data_bank_conflicts_pipe_lsu.sum

conflict_rate = bank_conflicts / shared_accesses * 100
```

**优化方向**：
```
如果 conflict_rate > 10%:
1. 添加 padding 到 shared memory 数组
   例如：__shared__ float data[32][33];  // +1 padding
2. 重新组织访问模式
3. 使用不同的 tile 大小
```

#### A4. Memory Bound 分析总结

**检查清单**：

| 指标 | 位置 | 理想值 | 问题诊断 |
|------|------|--------|---------|
| **DRAM 带宽利用率** | dram__throughput | < 90% | > 90% 表示达到极限 |
| **DRAM 实际带宽** | 手算 | > 80% 理论值 | < 80% 有优化空间 |
| **L2 命中率** | lts__t_sector_hit_rate | > 70% | < 70% 数据没复用 |
| **L1 命中率** | l1tex__t_sector_hit_rate | > 90% | < 90% cache 不友好 |
| **Coalesced 程度** | l1tex__average_t_sectors_per_request | = 1.0 | > 1.0 非合并访问 |
| **Bank Conflicts** | l1tex__data_bank_conflicts | = 0 | > 0 有冲突 |

### 路径 B：Compute-bound 深入分析

#### 命令

```bash
ncu --section ComputeWorkloadAnalysis \
    --section ComputeWorkloadAnalysis_Chart \
    -o compute_analysis ./program
```

#### B1. 查看指令吞吐量

**关键指标**：

```
# 整体指令吞吐量
smsp__inst_executed.avg.pct_of_peak_sustained_active

# 具体指令类型
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum     # FP32 加法
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum     # FP32 乘法
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum     # FP32 FMA
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum     # FP16 加法
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum     # FP16 乘法
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum     # FP16 FMA
```

**计算 FLOPs**：
```python
# FP32 FLOPs
fp32_flops = (
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum +
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum +
    2 * smsp__sass_thread_inst_executed_op_ffma_pred_on.sum  # FMA = 2 ops
)

# FP16 FLOPs
fp16_flops = (
    smsp__sass_thread_inst_executed_op_hadd_pred_on.sum +
    smsp__sass_thread_inst_executed_op_hmul_pred_on.sum +
    2 * smsp__sass_thread_inst_executed_op_hfma_pred_on.sum
)

# 计算吞吐量
duration_sec = gpu__time_duration.sum / 1e9
compute_throughput = fp32_flops / duration_sec  # FLOPS

# vs 理论峰值
peak_flops = 19.5e12  # A100 FP32 = 19.5 TFLOPS
efficiency = compute_throughput / peak_flops * 100
```

#### B2. 检查 Warp Divergence（分支发散）

**关键指标**：

```
smsp__thread_inst_executed_per_inst_executed.ratio

含义：平均每条指令有多少线程执行
理想值：32（整个 warp 都执行）
< 32: 有 divergence
```

**如何解读**：
```
= 32    →  ✅ 无 divergence，所有线程都执行
= 16    →  ❌ 平均只有一半线程执行（50% 浪费）
= 8     →  ❌ 只有 1/4 线程执行（75% 浪费）

divergence_waste = (32 - ratio) / 32 * 100
```

**示例分析**：
```
smsp__thread_inst_executed_per_inst_executed.ratio = 18.5

分析：
- 平均每条指令只有 18.5 个线程执行
- 浪费：(32 - 18.5) / 32 = 42% 的计算能力

可能原因：
if (threadIdx.x < threshold) {
    // 只有部分线程执行这里
    expensive_computation();
}

优化方向：
1. 重组数据，使 warp 内的线程走相同分支
2. 使用 __ballot_sync 等 warp 原语
3. 避免在 warp 内有不同的控制流
```

#### B3. 检查 ILP（指令级并行）

**关键指标**：

```
smsp__issue_active.avg.pct_of_peak_sustained_active
    # 指令发射利用率

smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
    # ALU 流水线利用率
```

**如何判断 ILP 不足**：
```
如果：
- SM Throughput 高（> 80%）
- 但 smsp__issue_active 低（< 60%）

说明：
- 有足够的计算，但指令发射受限
- ILP（指令级并行）不足

优化方向：
1. 每个线程处理多个独立数据（展开循环）
2. 使用向量化（float4）
3. 手动展开循环（#pragma unroll）
```

**示例优化**：
```cpp
// ❌ ILP 低
__global__ void low_ilp(float* out, float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = in[idx];
        a = a * 2.0f;
        a = a + 1.0f;
        a = a * a;
        out[idx] = a;
    }
}

// ✅ ILP 高
__global__ void high_ilp(float* out, float* in, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        // 4 个独立的计算可以并行
        float a0 = in[idx + 0];
        float a1 = in[idx + 1];
        float a2 = in[idx + 2];
        float a3 = in[idx + 3];

        a0 = a0 * 2.0f + 1.0f;
        a1 = a1 * 2.0f + 1.0f;
        a2 = a2 * 2.0f + 1.0f;
        a3 = a3 * 2.0f + 1.0f;

        a0 = a0 * a0;
        a1 = a1 * a1;
        a2 = a2 * a2;
        a3 = a3 * a3;

        out[idx + 0] = a0;
        out[idx + 1] = a1;
        out[idx + 2] = a2;
        out[idx + 3] = a3;
    }
}
```

### 路径 C：Occupancy 分析

#### 命令

```bash
ncu --section Occupancy \
    --section LaunchStats \
    -o occupancy_analysis ./program
```

#### C1. 占用率指标

**关键指标**：

```
sm__warps_active.avg.pct_of_peak_sustained_active
    # 实际占用率
    理想值：> 50%

sm__maximum_warps_per_active_cycle_pct
    # 理论最大占用率
```

#### C2. 占用率限制因素

**查看限制**：

在 NCU GUI 的 "Occupancy" section 中查看 "Limiting Factor"：

```
可能的限制因素：
1. Registers (寄存器)
2. Shared Memory (共享内存)
3. Block Size (线程块大小)
4. Warps per SM (每个 SM 的 warp 数)
```

**示例分析**：
```
Theoretical Occupancy: 50%
Achieved Occupancy:    45%
Limiting Factor:       Registers (128 per thread)

分析：
- 每个线程使用 128 个寄存器
- 限制了每个 SM 上的 block 数量
- 导致占用率只有 50%

优化方向：
1. 使用 __launch_bounds__ 限制寄存器
   __global__ void __launch_bounds__(256, 4) kernel(...)
2. 编译时限制：nvcc -maxrregcount=64
3. 简化 kernel 逻辑，减少寄存器使用
```

#### C3. Block Size 优化

**查看 Block Size 影响**：

```bash
# 使用 CUDA Occupancy Calculator
ncu --print-summary per-kernel ./program

# 或者在代码中使用
int blockSize;
int minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
```

### 路径 D：Warp State 分析

#### 命令

```bash
ncu --section WarpStateStats -o warp_analysis ./program
```

#### D1. Warp 停顿原因

**关键指标**：

```
Warp State 分布：

smsp__warps_active.avg.pct_of_peak_sustained_active
    # Active warps (正在执行)

smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active
    # Stalled on barrier (__syncthreads)

smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active
    # Stalled on long latency ops (memory)

smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active
    # Stalled on short latency ops (math)

smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active
    # Not selected for issue
```

**如何解读**：
```
如果 "stalled_long_scoreboard" > 50%:
    → Memory latency 是主要问题
    → 优化内存访问

如果 "stalled_barrier" > 30%:
    → __syncthreads() 太频繁
    → 减少同步点

如果 "stalled_short_scoreboard" > 30%:
    → 等待计算完成
    → 提高 ILP

如果 "not_selected" > 40%:
    → Occupancy 不足
    → 增加更多 warps
```

---

## 第三步：优化机会识别

### 查看 PTX/SASS 优化机会

#### 命令

```bash
# 启用源代码级分析
ncu --section SourceCounters \
    --section Scheduler \
    -o source_analysis ./program

# 需要编译时包含调试信息
nvcc -lineinfo -o program program.cu
```

#### 查看源代码热点

在 NCU GUI 中：
1. 打开 "Source" 页面
2. 选择 kernel
3. 查看每行代码的性能指标

**会显示**：
- 每行执行的指令数
- 每行的 stall cycles
- 内存访问次数
- 分支预测失败次数

#### 查看 SASS 代码

```bash
# 查看生成的 SASS 汇编
cuobjdump -sass program.o

# 或者在 NCU 中查看
# Details → Source → SASS tab
```

**查找优化机会**：

1. **查找多余的类型转换**
```sass
# ❌ 不好
FFMA R8, R4, R5, R6
I2F.F32.U32 R8, R8        ← 额外的转换！
F2I.U32.F32 R9, R8

# ✅ 好
FFMA R8, R4, R5, R6
```

2. **查找未融合的操作**
```sass
# ❌ 不好
FMUL R8, R4, R5
FADD R8, R8, R6           ← 应该用 FMA

# ✅ 好
FFMA R8, R4, R5, R6       ← FMA 一条指令
```

3. **查找分支代码**
```sass
# 如果看到很多
BRA label
@P0 BRA label
SYNC

说明有大量分支，考虑优化控制流
```

### 识别特定优化机会

#### 1. Tensor Core 机会

**指标**：
```
如果：
- Compute-bound
- 使用 FP16 或 FP32
- 矩阵乘法操作

检查：
smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active

= 0%: 没有使用 Tensor Cores
> 0%: 正在使用 Tensor Cores
```

**优化建议**：
```cpp
// 使用 WMMA API
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

wmma::load_matrix_sync(a_frag, A, K);
wmma::load_matrix_sync(b_frag, B, K);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
```

#### 2. Async Copy 机会 (Ampere+)

**指标**：
```
如果：
- Memory-bound
- GPU 是 Ampere 或更新
- 大量 Global → Shared memory 拷贝

检查：
smsp__inst_executed_op_global_ld.sum       # 全局加载
smsp__inst_executed_op_shared_st.sum       # 共享存储
```

**优化建议**：
```cpp
// Ampere+ 可以使用异步拷贝
#include <cuda/pipeline>

__global__ void async_copy_kernel(float* dst, const float* src, int N) {
    __shared__ float buffer[256];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    cuda::memcpy_async(buffer, src + blockIdx.x * 256, sizeof(buffer), pipe);
    pipe.producer_commit();
    pipe.consumer_wait();

    // 使用 buffer
    dst[threadIdx.x] = buffer[threadIdx.x] * 2.0f;
}
```

#### 3. 循环展开机会

**查看循环**：

在源代码视图中，如果看到循环：
```cpp
for (int i = 0; i < 8; i++) {
    result += data[i];
}
```

**检查是否展开**：

在 SASS 中，如果看到重复的指令（而不是循环），说明已展开。

**手动展开**：
```cpp
// 添加 pragma
#pragma unroll
for (int i = 0; i < 8; i++) {
    result += data[i];
}

// 或完全手动展开
result += data[0];
result += data[1];
result += data[2];
result += data[3];
result += data[4];
result += data[5];
result += data[6];
result += data[7];
```

---

## 完整分析检查清单

### 第一轮：快速诊断（5分钟）

```bash
ncu --section SpeedOfLight -o quick ./program
```

**检查项目**：

- [ ] SM Throughput
- [ ] Memory Throughput
- [ ] 判断是 Compute-bound 还是 Memory-bound

**决策**：
- Compute-bound → 跳到 Compute 分析
- Memory-bound → 跳到 Memory 分析
- Both Low → 跳到 Occupancy 分析

### 第二轮：深入分析（15分钟）

#### 如果是 Memory-bound：

```bash
ncu --section MemoryWorkloadAnalysis -o memory ./program
```

**检查项目**：

- [ ] DRAM 带宽利用率（目标 > 80%）
- [ ] L2 命中率（目标 > 70%）
- [ ] L1 命中率（目标 > 90%）
- [ ] Coalesced access（目标 = 1.0）
- [ ] Bank conflicts（目标 = 0）

#### 如果是 Compute-bound：

```bash
ncu --section ComputeWorkloadAnalysis -o compute ./program
```

**检查项目**：

- [ ] 指令吞吐量
- [ ] Warp divergence（目标 = 32）
- [ ] ILP 利用率
- [ ] 是否可用 Tensor Cores

#### 如果 Occupancy 低：

```bash
ncu --section Occupancy -o occupancy ./program
```

**检查项目**：

- [ ] 理论占用率
- [ ] 实际占用率
- [ ] 限制因素（Registers/Shared Memory/Block Size）
- [ ] Block Size 是否合理

### 第三轮：优化验证（10分钟）

**对比优化前后**：

```bash
# 优化前
ncu -o baseline ./program_old

# 优化后
ncu -o optimized ./program_new

# 对比
ncu-ui baseline.ncu-rep optimized.ncu-rep
```

**验证指标**：

- [ ] 运行时间是否减少
- [ ] 带宽利用率是否提高
- [ ] 缓存命中率是否提高
- [ ] Occupancy 是否提高
- [ ] Warp efficiency 是否提高

---

## 实战案例

### 案例 1：诊断并优化 Element-wise Kernel

#### Step 1: 快速诊断

```bash
ncu --section SpeedOfLight ./element_wise
```

**结果**：
```
SM Throughput:       12%
Memory Throughput:   88%
DRAM Throughput:     82%

→ Memory-bound
```

#### Step 2: Memory 深入分析

```bash
ncu --section MemoryWorkloadAnalysis ./element_wise
```

**结果**：
```
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld = 1.02
    → Coalesced access 很好！

lts__t_sector_hit_rate.pct = 15%
    → L2 命中率很低！数据没有复用

dram__bytes_read.sum = 800 MB
dram__bytes_write.sum = 400 MB
duration = 10 ms
实际带宽 = 120 GB/s
理论带宽 = 1555 GB/s (A100)
带宽利用率 = 7.7%
    → 远低于理论值！
```

#### Step 3: 优化方案

**分析**：
- Coalesced 已经很好
- 但带宽利用率很低
- 可能是多个小 kernel

**查看 kernel 调用**：
```bash
nsys profile -o timeline ./element_wise
```

**发现**：
```
有 3 个连续的 kernel：
1. relu_kernel     - 3.2 ms
2. add_bias_kernel - 3.5 ms
3. scale_kernel    - 3.1 ms

总共: 9.8 ms
```

**优化**：融合 kernel

```cpp
// 融合前：3 个 kernel
// 融合后：1 个 kernel
__global__ void fused_kernel(float* data, float bias, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        val = fmaxf(0.0f, val);  // ReLU
        val += bias;              // Add
        val *= scale;             // Scale
        data[idx] = val;
    }
}
```

#### Step 4: 验证

```bash
ncu --section SpeedOfLight ./element_wise_optimized
```

**结果**：
```
SM Throughput:       15%
Memory Throughput:   92%
DRAM Throughput:     87%
Duration:            3.5 ms

加速比：9.8ms / 3.5ms = 2.8x ✅
```

### 案例 2：诊断并优化 MatMul Kernel

#### Step 1: 快速诊断

```bash
ncu --section SpeedOfLight ./matmul
```

**结果**：
```
SM Throughput:       65%
Memory Throughput:   78%
DRAM Throughput:     72%

→ 两者都比较高，但都没达到极限
```

#### Step 2: Occupancy 分析

```bash
ncu --section Occupancy ./matmul
```

**结果**：
```
Theoretical Occupancy: 75%
Achieved Occupancy:    42%
Limiting Factor:       Registers (255 per thread)

Block Size: 256
Registers per thread: 255 (接近上限！)
```

#### Step 3: 优化

**方案 1：限制寄存器**

```cpp
__global__ void __launch_bounds__(256, 2)
matmul_kernel(...) {
    // kernel 代码
}
```

**方案 2：减小 Block Size**

```cpp
// 从 256 → 128
dim3 block(128);
```

#### Step 4: 验证

```bash
ncu --section Occupancy ./matmul_optimized
```

**结果**：
```
Theoretical Occupancy: 75%
Achieved Occupancy:    68% (从 42% 提升！)

Performance: 8.2 ms → 5.8 ms
加速比：1.4x ✅
```

---

## NCU 常用命令速查表

```bash
# 快速诊断
ncu --section SpeedOfLight ./program

# Memory 分析
ncu --section MemoryWorkloadAnalysis ./program

# Compute 分析
ncu --section ComputeWorkloadAnalysis ./program

# Occupancy 分析
ncu --section Occupancy ./program

# Warp 分析
ncu --section WarpStateStats ./program

# 源代码分析
ncu --section SourceCounters ./program

# 完整分析
ncu --set full -o profile ./program

# 特定 kernel
ncu --kernel-name mykernel ./program

# 导出 CSV
ncu --csv --page raw ./program > results.csv

# GUI 查看
ncu-ui profile.ncu-rep
```

---

## 总结：NCU 分析三步法

```
第一步：Speed of Light (5分钟)
├─ SM Throughput > 80%? → Compute-bound
├─ Memory Throughput > 80%? → Memory-bound
└─ Both < 50%? → Occupancy/其他问题

第二步：深入分析 (15分钟)
├─ Memory-bound
│  ├─ DRAM 带宽利用率
│  ├─ L2/L1 命中率
│  ├─ Coalesced access
│  └─ Bank conflicts
│
├─ Compute-bound
│  ├─ 指令吞吐量
│  ├─ Warp divergence
│  ├─ ILP 利用率
│  └─ Tensor Core 机会
│
└─ Occupancy 低
   ├─ 限制因素
   ├─ Block Size
   └─ 资源使用

第三步：优化验证 (10分钟)
└─ 对比优化前后，验证改进
```

**记住**：
- 不要试图一次性看所有指标
- 按照工作流逐步分析
- 每次优化后都要验证
- 使用 NCU GUI 更直观
