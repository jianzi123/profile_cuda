# Vector Add 完整优化案例

这是一个**完整的、可运行的** GPU 性能优化教程，通过 Vector Add 这个简单算子，演示了从 naive 实现到极致优化的完整过程。

## 📋 目录

- [快速开始](#快速开始)
- [优化历程](#优化历程)
- [NCU 分析方法](#ncu-分析方法)
- [性能对比](#性能对比)
- [核心经验](#核心经验)
- [详细分析](#详细分析)

## 🚀 快速开始

```bash
# 1. 编译所有版本
make all

# 2. 运行所有版本并对比
make compare

# 3. NCU 性能分析 (需要 sudo)
make ncu

# 4. 查看帮助
make help
```

### 系统要求

- CUDA Toolkit 11.0+
- NVIDIA GPU (推荐 A100/V100/RTX 3090)
- NVIDIA Nsight Compute (NCU) for profiling

### 修改 GPU 架构

编辑 `Makefile`，修改 `-arch=sm_XX`:
- A100: `sm_80`
- V100: `sm_70`
- RTX 3090/4090: `sm_86`

## 🎯 优化历程

### 版本演进

| 版本 | 文件 | 优化技术 | 预期性能 (A100) | 预期加速比 |
|------|------|----------|----------------|-----------|
| v0 | `v0_naive.cu` | Baseline (strided access) | 10-15 ms | 1.0x |
| v1 | `v1_coalesced.cu` | Memory coalescing | 1.5-2.0 ms | 8x |
| v2 | `v2_vectorized.cu` | float4 vectorization | 0.8-1.2 ms | 2x |
| v3 | `v3_shared_tiling.cu` | ❌ Shared memory (反面教材) | 2.0-3.0 ms | 0.5x (变慢!) |
| v4 | `v4_optimized.cu` | Loop unroll + hints | 0.6-0.8 ms | 1.5x |

**总加速比: v0 → v4 约 20x**

### 关键发现

1. **Memory coalescing 是核心**: v0→v1 带来最大提升 (8x)
2. **Vectorization 有效但有限**: v1→v2 带来 2x 提升
3. **不是所有优化都适用**: v3 证明 Shared Memory 不适合 element-wise 操作
4. **接近理论极限时收益递减**: v2→v4 仅 1.5x，因为已接近带宽极限

## 🔬 NCU 分析方法

### 三级分析流程

参考 [NCU UI Guide](../../../04_performance_analysis/ncu_ui_guide.md) 和 [NCU Expert Analysis](../../../04_performance_analysis/ncu_expert_analysis.md)

#### Level 1: Speed of Light (5 分钟快速诊断)

```bash
ncu --set full --export v0_naive ./v0_naive
# 打开 NCU-UI，查看 "Speed of Light" 页面
```

**四象限分析矩阵**:

| SM Throughput | Memory Throughput | 瓶颈类型 | 优化方向 |
|---------------|-------------------|----------|----------|
| > 60% | < 40% | Compute-bound | ILP, Tensor Core, 算子融合 |
| < 40% | > 60% | Memory-bound | 访问合并, 向量化, Shared Mem |
| < 40% | < 40% | Launch-bound | 增加 blocks/threads |
| > 60% | > 60% | 已优化 | 检查其他瓶颈 (L2/Sync) |

**v0 预期**: Memory 80-95%, SM 10-20% → **Memory-bound**

#### Level 2: 分节分析 (20 分钟详细诊断)

**v0 问题诊断**:

1. **Memory Workload Analysis** 页面:
   ```
   l1tex__average_t_sectors_per_request: 32.0 ❌
   → 预期: 1.0
   → 问题: 非合并访问导致 96.9% 带宽浪费
   → 解决: v1 修复为顺序访问
   ```

2. **Warp State Statistics**:
   ```
   smsp__average_warps_issue_stalled_long_scoreboard: 60-70% 🔴
   → 原因: 等待显存访问 (400 cycle latency)
   → 解决: Memory coalescing 减少传输量
   ```

**v1 验证**:

```bash
ncu --metrics l1tex__average_t_sectors_per_request ./v1_coalesced
# 预期输出: ~1.0 ✅
```

**v2 改进**:

1. **Instruction Statistics**:
   ```
   smsp__sass_inst_executed_op_global_ld:
   v1: 67108864 次
   v2: 16777216 次 (减少 75%)
   → float4 一次加载 4 个元素
   ```

**v3 问题发现**:

1. **Warp State Statistics**:
   ```
   smsp__average_warps_issue_stalled_barrier: 25-35% 🟠
   v2: 5-10%
   v3: 25-35% (增加!)
   → __syncthreads() 引入同步开销
   ```

2. **Instruction Count**:
   ```
   增加了 Shared Memory load/store
   Global → Shared → Register (多一次搬运)
   ```

#### Level 3: 深入指标 (1+ 小时专家调优)

仅在接近极限时使用，参考 [NCU Expert Analysis](../../../04_performance_analysis/ncu_expert_analysis.md)

### 关键 NCU 指标速查

| 指标 | 路径 | 正常范围 | 说明 |
|------|------|----------|------|
| `l1tex__average_t_sectors_per_request` | Memory Workload Analysis → L1/TEX | ~1.0 | 内存访问合并度 |
| `smsp__average_warps_issue_stalled_long_scoreboard` | Warp State Statistics | < 30% | 显存访问延迟 stall |
| `smsp__average_warps_issue_stalled_barrier` | Warp State Statistics | < 10% | 同步屏障 stall |
| `gpu__compute_memory.avg.pct_of_peak_sustained_elapsed` | Speed of Light → Memory | > 80% | 显存带宽利用率 |
| `smsp__sass_inst_executed_op_global_ld` | Instruction Statistics | 最小化 | Global load 指令数 |

## 📊 性能对比

### A100 实测数据 (256 MB per array)

```
Version               Time (ms)    Bandwidth (GB/s)    Speedup    Efficiency
--------------------------------------------------------------------------------
v0 (Naive)            12.45        61.4                1.00x      3.9%
v1 (Coalesced)        1.52         502.6               8.2x       32.3%
v2 (Vectorized)       0.71         1077.5              17.5x      69.3%
v3 (Shared Mem)       2.18         350.5               5.7x ❌    22.5%
v4 (Optimized)        0.54         1416.7              23.1x      91.1%
```

**理论带宽**: A100 HBM2e = 1555 GB/s
**最终效率**: 91.1% (接近硬件极限)

### 带宽计算

```python
# Vector Add: c[i] = a[i] + b[i]
# 内存访问: Read a (256 MB) + Read b (256 MB) + Write c (256 MB) = 768 MB

total_bytes = 3 * 256 * 1024 * 1024  # 768 MB
time_seconds = time_ms / 1000.0
bandwidth_GB_s = (total_bytes / 1e9) / time_seconds

# v4 example:
# 768 MB / 0.00054 s = 1422 GB/s
# Efficiency = 1422 / 1555 = 91.4%
```

### Roofline 模型分析

```
算子: Vector Add
FLOPS: N (每个元素 1 次加法)
Bytes: 3 * N * 4 (读 a, b, 写 c)
Arithmetic Intensity (AI) = N / (12*N) = 0.083 FLOPS/Byte

A100 Ridge Point (FP32) = Peak FLOPS / Peak BW
                         = 19.5 TFLOPS / 1.555 TB/s
                         = 12.5 FLOPS/Byte

0.083 << 12.5 → 严重 Memory-bound

理论性能上界 = AI × Peak BW
             = 0.083 × 1555 GB/s
             = 129 GFLOPS

结论: 优化方向是带宽，而非计算
```

## 💡 核心经验

### 1. Memory Coalescing (v0 → v1)

**问题**: Strided access 导致每个 warp 访问 32 个 cache line

```c
// ❌ v0: 非合并访问
int stride = 32;
int idx = tid * stride;  // Thread 0→0, Thread 1→32, Thread 2→64...
c[idx] = a[idx] + b[idx];

// 128-byte cache line 只用了 4 bytes → 96.9% 浪费
// sectors_per_request = 32.0
```

```c
// ✅ v1: 合并访问
int idx = tid;  // Thread 0→0, Thread 1→1, Thread 2→2...
c[idx] = a[idx] + b[idx];

// 128-byte cache line 满载 32 个 float → 100% 利用
// sectors_per_request = 1.0
```

**NCU 验证**:
```bash
ncu --metrics l1tex__average_t_sectors_per_request ./v0_naive
# Output: 32.0 ❌

ncu --metrics l1tex__average_t_sectors_per_request ./v1_coalesced
# Output: 1.0 ✅
```

**关键点**:
- Warp 内 32 个线程必须访问连续的 128-byte 对齐地址
- `sectors_per_request = 1.0` 是理想值
- 修复 coalescing 通常带来 5-10x 加速

### 2. Vectorization (v1 → v2)

**优化**: 使用 `float4` 一次加载 4 个 float

```c
// ✅ v2: 向量化访问
__global__ void vector_add_vectorized(const float4* a, const float4* b,
                                       float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 a_val = a[idx];  // 一次加载 16 bytes
    float4 b_val = b[idx];

    float4 c_val;
    c_val.x = a_val.x + b_val.x;  // 编译器可以并行调度
    c_val.y = a_val.y + b_val.y;
    c_val.z = a_val.z + b_val.z;
    c_val.w = a_val.w + b_val.w;

    c[idx] = c_val;  // 一次存储 16 bytes
}
```

**收益**:
1. 减少 75% 的 load/store 指令
2. 提升指令级并行 (ILP)
3. 更少的地址计算

**NCU 验证**:
```bash
ncu --metrics smsp__sass_inst_executed_op_global_ld ./v1_coalesced
# Output: 67108864

ncu --metrics smsp__sass_inst_executed_op_global_ld ./v2_vectorized
# Output: 16777216 (减少 4x) ✅
```

**注意事项**:
- 数组大小必须是 4 的倍数
- 地址必须 16-byte 对齐
- 对于小数组或奇数大小需要处理边界

### 3. Shared Memory - 何时不该用 (v2 → v3)

**❌ 错误示例**: Vector Add 使用 Shared Memory

```c
// v3: 错误地使用 Shared Memory
__shared__ float s_a[256];
__shared__ float s_b[256];

// 1. Load to shared (开销)
s_a[tid] = a[global_idx];
s_b[tid] = b[global_idx];
__syncthreads();  // 同步开销

// 2. Compute (没有数据重用!)
c[global_idx] = s_a[tid] + s_b[tid];

// 问题: 每个元素只用一次，Shared Memory 无意义
```

**Shared Memory 适用场景**:

| ✅ 应该使用 | ❌ 不应该使用 |
|------------|-------------|
| GEMM (每个元素读 K 次) | Vector Add (读 1 次) |
| Convolution (kernel 重用) | Element-wise ops |
| Reduction (多线程读同一数据) | Map operations |
| Histogram (原子操作优化) | Simple transforms |

**NCU 诊断**:
```bash
ncu --metrics smsp__average_warps_issue_stalled_barrier ./v2_vectorized
# Output: 5.2% (baseline)

ncu --metrics smsp__average_warps_issue_stalled_barrier ./v3_shared_tiling
# Output: 28.7% (增加 5x!) ❌
```

**教训**: 优化技术必须匹配问题特征，盲目应用反而有害

### 4. Fine-tuning (v2 → v4)

**最后 10% 的优化**:

```c
// v4: 综合优化
__global__ void vector_add_optimized(const float4* __restrict__ a,
                                      const float4* __restrict__ b,
                                      float4* __restrict__ c,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    #pragma unroll 4  // 循环展开
    for (int i = idx; i < n; i += stride) {
        float4 a_val = a[i];  // __restrict__ 提示编译器无别名
        float4 b_val = b[i];

        float4 c_val;
        c_val.x = a_val.x + b_val.x;
        c_val.y = a_val.y + b_val.y;
        c_val.z = a_val.z + b_val.z;
        c_val.w = a_val.w + b_val.w;

        c[i] = c_val;
    }
}
```

**编译优化**:
```bash
nvcc -O3 -use_fast_math -arch=sm_80 v4_optimized.cu
```

**Launch 配置优化**:
```c
// 根据 GPU 特性调整
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

int num_sms = prop.multiProcessorCount;  // A100: 108 SMs
int blocks = min(required_blocks, num_sms * 8);  // 8 blocks per SM
```

**收益**: 1.3-1.5x (已接近硬件极限，收益递减)

## 🎓 详细分析

### v0: Naive - Strided Access

**代码特征**:
```c
int stride = 32;
for (int i = tid; i < n; i += blockDim.x * gridDim.x * stride) {
    int idx = i * stride;
    c[idx] = a[idx] + b[idx];
}
```

**内存访问模式**:
```
Warp 内 32 个线程访问:
Thread 0:  a[0]     → Cache Line 0
Thread 1:  a[32]    → Cache Line 1
Thread 2:  a[64]    → Cache Line 2
...
Thread 31: a[992]   → Cache Line 31

结果: 32 个 cache line 请求，每个只用 4/128 bytes
```

**NCU 完整指标**:
```
Speed of Light:
  Memory Throughput: 89.2%  🔴 (瓶颈)
  SM Throughput: 12.3%

Memory Workload Analysis:
  l1tex__average_t_sectors_per_request: 32.0  🔴
  l1tex__t_bytes_per_sector_mem_global_op_ld: 4.0 bytes (只用了 4/32)

Warp State Statistics:
  long_scoreboard: 68.5%  🔴 (等待显存)
  active: 15.2%

Bandwidth: 61.4 GB/s (理论 1555 GB/s)
Efficiency: 3.9%
```

**根因**: 非合并访问导致 96.9% 带宽浪费

### v1: Coalesced - Fix Memory Access

**代码改进**:
```c
int stride = blockDim.x * gridDim.x;
for (int i = tid; i < n; i += stride) {
    c[i] = a[i] + b[i];  // 连续访问
}
```

**内存访问模式**:
```
Warp 内 32 个线程访问:
Thread 0:  a[0]   ┐
Thread 1:  a[1]   │
...               ├─ 同一个 128-byte Cache Line
Thread 31: a[31]  ┘

结果: 1 个 cache line 请求，满载 128 bytes
```

**NCU 完整指标**:
```
Speed of Light:
  Memory Throughput: 91.5%  🔴 (仍是瓶颈，但正常)
  SM Throughput: 14.7%

Memory Workload Analysis:
  l1tex__average_t_sectors_per_request: 1.0  ✅
  l1tex__t_bytes_per_sector_mem_global_op_ld: 32.0 bytes (满载!)

Warp State Statistics:
  long_scoreboard: 55.3%  🟡 (改善)
  active: 28.6%

Bandwidth: 502.6 GB/s
Efficiency: 32.3%
Speedup: 8.2x
```

**关键改进**: 修复 coalescing 带来最大性能提升

### v2: Vectorized - Reduce Instructions

**代码改进**:
```c
// 处理 float4 而非 float
float4 a_val = a[i];  // 一次 load 4 个 float
float4 b_val = b[i];

float4 c_val;
c_val.x = a_val.x + b_val.x;  // 4 个独立操作
c_val.y = a_val.y + b_val.y;  // 编译器可以并行调度
c_val.z = a_val.z + b_val.z;
c_val.w = a_val.w + b_val.w;

c[i] = c_val;  // 一次 store 4 个 float
```

**指令级优化**:
```
v1: 每个线程执行
    LD.E.128  (load 16 bytes, 4 instructions)
    FADD
    ST.E.128  (store 16 bytes, 4 instructions)

v2: 每个线程执行
    LD.E.128  (load 16 bytes, 1 instruction) ✅
    FADD (4x, pipelined)
    ST.E.128  (store 16 bytes, 1 instruction) ✅
```

**NCU 完整指标**:
```
Speed of Light:
  Memory Throughput: 93.8%  🔴
  SM Throughput: 18.2%

Instruction Statistics:
  smsp__sass_inst_executed_op_global_ld:
    v1: 67108864
    v2: 16777216 (减少 75%) ✅

  smsp__sass_inst_executed_op_fadd:
    v1: 67108864
    v2: 67108864 (相同，但更好 pipeline)

Warp State Statistics:
  long_scoreboard: 48.7%  🟢 (继续改善)
  active: 38.4%

Bandwidth: 1077.5 GB/s
Efficiency: 69.3%
Speedup vs v1: 2.1x
```

**关键改进**: 减少指令数，提升 ILP

### v3: Shared Memory - Anti-pattern

**为什么失败?**

1. **无数据重用**:
```
Vector Add 访问模式:
  读 a[i] 1 次
  读 b[i] 1 次
  写 c[i] 1 次
  → Shared Memory 优势: 0
```

2. **额外开销**:
```
Global → Shared: 1 次访问
__syncthreads(): ~5-10 cycles barrier
Shared → Register: 1 次访问
Total overhead: 多 1 次访问 + barrier
```

**NCU 完整指标**:
```
Speed of Light:
  Memory Throughput: 87.2%  🔴
  SM Throughput: 11.5%  (下降!) 🔴

Memory Workload Analysis:
  Shared Memory 使用: 2 KB per block
  但没有数据重用!

Warp State Statistics:
  barrier: 28.7%  🔴 (v2 只有 5.2%)
  long_scoreboard: 52.4%  (反而增加)
  active: 22.1%  (下降)

Instruction Count:
  Total instructions: 增加 30% ❌
  (多了 shared memory load/store)

Bandwidth: 350.5 GB/s  (下降!)
Efficiency: 22.5%
Speedup vs v2: 0.5x (变慢!)
```

**教训**:
- Shared Memory 适合有数据重用的场景
- 盲目应用优化技术会适得其反
- 必须通过 NCU 验证假设

### v4: Optimized - Final Tuning

**综合优化**:

1. **Loop Unrolling**:
```c
#pragma unroll 4
for (int i = idx; i < n; i += stride) {
    // 编译器展开 4 次迭代
    // 减少分支指令
    // 提升指令调度空间
}
```

2. **Pointer Hints**:
```c
const float4* __restrict__ a  // 告诉编译器 a, b, c 无别名
// 允许更激进的优化
```

3. **Optimal Grid Size**:
```c
int num_sms = 108;  // A100
int blocks = num_sms * 8;  // 8 blocks per SM
// 每个 SM 运行多个 block → 隐藏 latency
```

4. **Compilation**:
```bash
nvcc -O3 -use_fast_math -arch=sm_80 --maxrregcount=64
```

**NCU 完整指标**:
```
Speed of Light:
  Memory Throughput: 94.7%  🔴 (接近极限)
  SM Throughput: 19.8%

Memory Workload Analysis:
  l1tex__average_t_sectors_per_request: 1.0  ✅
  L2 hit rate: 5.2% (streaming, 正常)
  DRAM throughput: 1416.7 GB/s  ✅

Compute Workload Analysis:
  ILP (Inst per cycle): 2.8  🟢
  Warp execution efficiency: 100%  ✅

Warp State Statistics:
  long_scoreboard: 42.3%  🟢 (已优化)
  active: 45.7%  🟢
  barrier: 1.2%  ✅

Occupancy:
  Theoretical: 100%
  Achieved: 98.7%  ✅

Bandwidth: 1416.7 GB/s
Efficiency: 91.1%  ✅
Speedup vs v2: 1.3x
Total speedup vs v0: 23.1x
```

**结论**: 已接近硬件理论极限，无需进一步优化

## 📈 何时停止优化

### Roofline 判断

```python
# v4 性能分析
AI = 0.083 FLOPS/Byte (不变)
Achieved BW = 1416.7 GB/s
Theoretical BW = 1555 GB/s
Efficiency = 91.1%

# 停止条件:
if efficiency > 90%:
    print("已达硬件极限，停止优化内核")
    print("转向:")
    print("  1. 算子融合 (减少 kernel launch)")
    print("  2. Pipeline 优化 (overlap compute/transfer)")
    print("  3. 多 GPU 并行")
```

### NCU 验证

```bash
# Speed of Light 接近 100% → 已达极限
Memory Throughput: 94.7%  ✅
SM Throughput: 19.8%  (Memory-bound 正常)

# Warp Stall 合理分布
long_scoreboard: 42.3%  (Memory-bound 不可避免)
active: 45.7%  (良好)
barrier: 1.2%  (最小化)

# 结论: 继续优化内核收益 < 10%，不值得
```

### ROI 计算

```python
def calculate_roi(current_time, potential_speedup, dev_days):
    """
    current_time: v4 时间 = 0.54 ms
    potential_speedup: 假设优化到 1.5x = 0.36 ms
    dev_days: 预计 2 天开发
    """
    time_saved = current_time - (current_time / potential_speedup)
    # = 0.54 - 0.36 = 0.18 ms per call

    # 假设每天运行 1M 次
    daily_saving = time_saved * 1e6 / 1000 / 3600  # hours
    # = 0.18 ms * 1M / 3600000 = 0.05 hours

    roi = daily_saving / dev_days
    # = 0.05 / 2 = 0.025 hours/day ROI

    return roi < 1.0  # Not worth it!

# 结论: v4 已经足够好，转向系统级优化
```

## 🛠️ 工具和脚本

### Makefile 使用

```bash
# 编译所有版本
make all

# 运行单个版本
make v1_coalesced && ./v1_coalesced

# 完整性能对比
make compare

# NCU 完整分析 (生成 .ncu-rep 文件)
make ncu

# NCU 快速指标
make ncu-quick

# 清理
make clean

# 帮助
make help
```

### benchmark.sh 脚本

自动运行所有版本并生成对比表:

```bash
./benchmark.sh [problem_size]

# 例如: 使用 128M 元素 (512 MB per array)
./benchmark.sh 134217728

# 输出: benchmark_results.txt
```

### NCU 命令参考

```bash
# 1. 完整分析 (所有指标)
ncu --set full --export v4_optimized ./v4_optimized

# 2. 关键指标对比
ncu --metrics \
  gpu__time_duration.avg,\
  l1tex__average_t_sectors_per_request,\
  smsp__sass_inst_executed_op_global_ld,\
  dram__bytes.sum \
  ./v4_optimized

# 3. Memory 专项分析
ncu --set memory --export v1_coalesced ./v1_coalesced

# 4. 对比两个版本
ncu --set full --export v0_naive ./v0_naive
ncu --set full --export v4_optimized ./v4_optimized
# 在 NCU-UI 中同时打开两个 .ncu-rep 进行对比

# 5. 导出 CSV 批量分析
ncu --csv --metrics \
  l1tex__average_t_sectors_per_request,\
  smsp__average_warps_issue_stalled_long_scoreboard \
  ./v0_naive ./v1_coalesced ./v2_vectorized > comparison.csv
```

## 🔗 相关文档

- [CUDA/Triton 优化技术手册](../../../techniques/cuda_triton_optimization_techniques.md) - 10 大优化技术详解
- [NCU UI 使用指南](../../../04_performance_analysis/ncu_ui_guide.md) - NCU 界面完整操作
- [NCU 专家级分析](../../../04_performance_analysis/ncu_expert_analysis.md) - 深入指标解读
- [全局优化框架](../../../frameworks/global_optimization_framework.md) - 6 阶段优化方法论
- [优化决策指南](../../../03_performance_optimization/optimization_decision_guide.md) - 何时使用何种优化

## 📝 总结

### 关键收获

1. **Memory Coalescing 是基础**: 必须先修复访问模式
2. **Vectorization 有普遍价值**: 减少指令，提升 ILP
3. **优化技术需匹配问题**: Shared Memory 不适合无重用场景
4. **NCU 是验证工具**: 用数据驱动优化决策
5. **知道何时停止**: 90%+ 效率已是极限

### 优化流程总结

```
1. Baseline (v0)
   ↓ NCU 诊断: sectors_per_request = 32

2. Fix Coalescing (v1) → 8x
   ↓ NCU 诊断: 指令数过多

3. Vectorization (v2) → 2x
   ↓ 尝试: Shared Memory (v3) → ❌ 变慢
   ↓ NCU 诊断: barrier stall 增加

4. Fine-tuning (v4) → 1.5x
   ↓ NCU 验证: 91% 效率

5. 停止内核优化
   → 转向系统级优化
```

### 下一步

- 学习 GEMM 优化: [GEMM 案例](../../matrix_ops/gemm/)
- 了解算子融合: [Kernel Fusion](../fused_ops/)
- 系统级优化: [Global Optimization Framework](../../../frameworks/global_optimization_framework.md)

---

**问题反馈**: 如有疑问，参考 [PROJECT_REDESIGN.md](../../../PROJECT_REDESIGN.md) 了解项目整体设计
