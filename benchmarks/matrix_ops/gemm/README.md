# GEMM 完整优化案例

Matrix Multiplication (GEMM - General Matrix Multiply) 是最重要的 GPU 计算核心，广泛应用于深度学习、科学计算等领域。

本案例展示从 naive 实现到优化版本的完整优化历程，性能提升 **30-40x**。

## 🎯 优化目标

**任务**: 计算 C = A × B
- A: M × K 矩阵
- B: K × N 矩阵
- C: M × N 矩阵

**计算量**: 2MNK FLOPs (每个输出元素: K 次乘法 + K 次加法)

**内存访问**: (MK + KN + MN) × 4 bytes

## 📋 版本演进

| 版本 | 优化技术 | 预期性能 (A100) | 加速比 | 代码复杂度 |
|------|----------|----------------|--------|-----------|
| v0 | Naive (全局内存) | ~150 GFLOPS | 1.0x | ⭐ |
| v2 | Shared Memory Tiling | ~2500 GFLOPS | 15x | ⭐⭐⭐ |
| v3 | Bank Conflict + Unroll | ~5000 GFLOPS | 2x | ⭐⭐⭐⭐ |
| Triton | 自动优化 | ~4000 GFLOPS | - | ⭐ |
| cuBLAS | NVIDIA 官方库 | ~15000 GFLOPS | 3x | ⭐ (API调用) |

**理论峰值**: A100 FP32 = 19,500 GFLOPS

## 🚀 快速开始

```bash
# 编译所有版本
make all

# 运行对比 (1024×1024×1024)
make run

# NCU 性能分析
make ncu

# Python 工具对比
make compare
```

## 📊 优化历程详解

### v0: Naive Implementation

**代码特征**:
```c
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
}
```

**问题分析**:

1. **显存访问重复严重**:
   - A 的每个元素被读 N 次
   - B 的每个元素被读 M 次
   - 总显存流量: (MK × N + KN × M) × 4 bytes

   示例 (1024×1024×1024):
   ```
   A: 4 MB, 但实际读取 4 MB × 1024 = 4 GB
   B: 4 MB, 但实际读取 4 MB × 1024 = 4 GB
   总计: 8 GB (重复 1024 倍!)
   ```

2. **非合并访问 B 矩阵**:
   - B 按列访问: `B[k * N + col]`
   - stride = N, 非连续
   - `sectors_per_request` >> 1.0

3. **Arithmetic Intensity 极低**:
   ```
   AI = 2K FLOPs / (2K × 4 bytes) = 0.25 FLOPS/Byte
   Ridge Point (A100) = 12.5 FLOPS/Byte
   → 严重 Memory-bound
   ```

**NCU 诊断**:
```bash
ncu --metrics \
  l1tex__average_t_sectors_per_request,\
  dram__bytes.sum,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./gemm_v0_naive 1024 1024 1024

预期结果:
  - sectors_per_request: 16-32 (B 列访问)
  - dram__bytes.sum: ~8 GB (重复读取)
  - SM Throughput: 5-10% (计算单元闲置)
  - Memory Throughput: 85-95% (显存瓶颈)
```

**性能**: ~150 GFLOPS (0.8% 峰值)

### v2: Shared Memory Tiling

**核心思想**: 分块加载到 Shared Memory，实现数据重用

**Tiling 原理**:

```
原始计算: C[i][j] = Σ A[i][k] × B[k][j]

分块计算:
  将 K 维度分成 tiles: K = TILE_SIZE × num_tiles

  for each tile_k:
      1. 加载 A_tile[TILE_SIZE × TILE_SIZE] 到 Shared Memory
      2. 加载 B_tile[TILE_SIZE × TILE_SIZE] 到 Shared Memory
      3. 计算 partial sum (重用 tile 数据 TILE_SIZE 次)
      4. 累加到结果
```

**数据重用分析** (TILE_SIZE = 32):

```
每个 tile:
  - A_tile 加载 1 次, 使用 32 次 (每个 row 被 32 个 column 使用)
  - B_tile 加载 1 次, 使用 32 次 (每个 column 被 32 个 row 使用)

显存流量:
  - 原来: 8 GB
  - 现在: (MK + KN) × 4 bytes = 8 MB
  - 减少: 1000x!
```

**Arithmetic Intensity 提升**:

```
每读取 1 float (4 bytes) 做 TILE_SIZE = 32 次计算
AI = 32 / 4 = 8.0 FLOPS/Byte

虽然仍 < Ridge Point (12.5), 但已大幅改善
```

**代码要点**:

```c
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

for (int t = 0; t < num_tiles; t++) {
    // Collaborative loading (256 threads 并行加载)
    As[ty][tx] = A[...];
    Bs[ty][tx] = B[...];
    __syncthreads();  // 确保 tile 加载完成

    // Compute using shared memory (数据重用!)
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();  // 确保计算完成再加载下一个 tile
}
```

**NCU 验证**:

```bash
ncu --metrics \
  dram__bytes.sum,\
  l1tex__data_pipe_lsu_wavefronts_mem_shared,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./gemm_v2_shared_tiling 1024 1024 1024

预期改进:
  - dram__bytes.sum: 从 8 GB → 8 MB (1000x 减少) ✅
  - Shared Memory 访问激增 (数据重用证据)
  - SM Throughput: 提升到 25-35%
  - 性能: ~2500 GFLOPS (15x vs v0)
```

**仍然存在的问题**:

1. **Bank Conflict**:
   ```c
   As[ty][k]  // 列访问, 同一 column 的不同 row 可能冲突
   Bs[k][tx]  // 行访问, 同一 row 的不同 column 可能冲突
   ```

2. **非向量化访问**: 单个 float 加载

3. **ILP 不足**: 每个线程只计算一个输出元素

### v3: Optimized (Bank Conflict Fix + Fine-tuning)

**优化 1: Bank Conflict Resolution**

**问题**: Shared Memory 有 32 个 bank, 同时访问同一 bank 会串行化

```c
// ❌ 可能有 bank conflict
__shared__ float As[32][32];

As[0][0] → Bank 0
As[1][0] → Bank 0  // 冲突! (如果多个线程同时访问)
As[2][0] → Bank 0

// ✅ Padding 避免 conflict
__shared__ float As[32][33];  // +1 padding

As[0][0] → Bank 0
As[1][0] → Bank (33 % 32) = 1  // 不同 bank!
As[2][0] → Bank (66 % 32) = 2
```

**原理**: 每行多 1 个元素, 列访问时自动错开 bank

**NCU 验证**:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld \
  ./gemm_v2_shared_tiling

# v2: bank_conflicts > 1000
# v3: bank_conflicts ≈ 0 ✅
```

**优化 2: Loop Unrolling**

```c
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++) {
    sum += As[ty][k] * Bs[k][tx];
}

编译器展开 →
sum += As[ty][0] * Bs[0][tx];
sum += As[ty][1] * Bs[1][tx];
...
sum += As[ty][31] * Bs[31][tx];
```

**收益**:
- 减少循环分支指令
- 提升 ILP (多个 FMA 并行)
- 预期提升: 1.1-1.2x

**优化 3: Register Tiling** (高级)

每个线程计算多个输出元素 (如 4×4 block):

```c
float sum[4][4];  // 16 个累加器

// 每个线程处理 4×4 = 16 个输出
for (int k = 0; k < TILE_SIZE; k++) {
    float a[4], b[4];  // 寄存器

    // Load 4 elements
    a[0] = As[ty*4 + 0][k];
    a[1] = As[ty*4 + 1][k];
    a[2] = As[ty*4 + 2][k];
    a[3] = As[ty*4 + 3][k];

    b[0] = Bs[k][tx*4 + 0];
    b[1] = Bs[k][tx*4 + 1];
    b[2] = Bs[k][tx*4 + 2];
    b[3] = Bs[k][tx*4 + 3];

    // Outer product
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            sum[i][j] += a[i] * b[j];
}
```

**收益**: 减少 Shared Memory 访问, 提升 ILP

**综合性能**: ~5000 GFLOPS (25% 峰值, 2x vs v2)

### Triton vs CUDA 对比

**Triton 优势**:
- 代码量 1/3
- 自动 tiling, shared memory 管理, bank conflict 避免
- 接近 CUDA v3 性能

**CUDA 优势**:
- 完全控制 (极致优化可达 50-60% 峰值)
- Tensor Core 支持更好 (WMMA, cuBLAS)
- 更多底层优化技巧 (warp specialization, async copy, etc.)

## 🏆 极致优化方向

### 1. Tensor Core (最重要!)

A100 Tensor Core 性能:
- FP32: 19.5 TFLOPS (标量)
- TF32: 156 TFLOPS (Tensor Core, 自动降精度)
- FP16: 312 TFLOPS (Tensor Core)

**使用方法**:

**cuBLAS** (最简单):
```c
cublasHandle_t handle;
cublasCreate(&handle);

// 自动使用 Tensor Core
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha, B, CUDA_R_32F, N,
            A, CUDA_R_32F, K,
    &beta,  C, CUDA_R_32F, N,
    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

**WMMA API** (手动控制):
```c
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

load_matrix_sync(a_frag, A, K);
load_matrix_sync(b_frag, B, K);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C, c_frag, N, mem_row_major);
```

**预期性能**: 10000-15000 GFLOPS (50-80% Tensor Core 峰值)

### 2. 其他高级技巧

**Double Buffering**:
```c
// Overlap compute 和 memory load
__shared__ float As[2][TILE_SIZE][TILE_SIZE+1];
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE+1];

for (int t = 0; t < num_tiles; t++) {
    int load_idx = t % 2;
    int compute_idx = (t + 1) % 2;

    // Load next tile while computing current tile
    if (t < num_tiles - 1) {
        load_tile_async(As[load_idx], Bs[load_idx], t+1);
    }

    // Compute current tile
    compute(As[compute_idx], Bs[compute_idx]);
}
```

**Warp Specialization**:
- 部分 warp 专门做数据加载
- 部分 warp 专门做计算
- 提升 overlap 程度

## 📈 性能对比总结 (1024×1024×1024)

| 版本 | 时间 (ms) | GFLOPS | 加速比 | 效率 |
|------|-----------|--------|--------|------|
| v0 Naive | 14.3 | 150 | 1.0x | 0.8% |
| v2 Shared Tiling | 0.86 | 2500 | 16.6x | 12.8% |
| v3 Optimized | 0.43 | 5000 | 2.0x | 25.6% |
| **cuBLAS (TF32)** | 0.14 | **15000** | 3.0x | **96%** |

**Roofline 分析**:

```
AI 分析:
  v0: AI = 0.25 → 严重 Memory-bound
  v2: AI = 8.0  → 仍 Memory-bound 但改善
  v3: AI = 8.0  → 接近 Ridge Point

Tensor Core Ridge Point (TF32):
  = 156 TFLOPS / 1.555 TB/s = 100 FLOPS/Byte

  → 使用 Tensor Core 后变成 Compute-bound
  → 性能跳跃式提升!
```

## 🔧 NCU 完整分析流程

### Step 1: 快速诊断 (Speed of Light)

```bash
ncu --set full --export gemm_v0 ./gemm_v0_naive 1024 1024 1024
```

打开 NCU-UI → Speed of Light:
- Memory Throughput: 90% → Memory-bound
- SM Throughput: 8% → 计算单元闲置

**结论**: 优化方向 = 减少显存访问

### Step 2: 定位问题 (Memory Workload Analysis)

关键指标:
```
l1tex__average_t_sectors_per_request: 32.0
  → B 矩阵列访问非合并

dram__bytes.sum: 8 GB
  → 重复读取严重 (实际只需 8 MB)

l1tex__t_bytes_per_sector_mem_global_op_ld: 4 bytes
  → 每个 sector 只用 4/32 = 12.5%
```

**解决方案**: Shared Memory Tiling

### Step 3: 验证优化 (v2)

```bash
ncu --set full --export gemm_v2 ./gemm_v2_shared_tiling 1024 1024 1024
```

对比指标:
```
dram__bytes.sum:
  v0: 8 GB
  v2: 8 MB ✅ (减少 1000x)

SM Throughput:
  v0: 8%
  v2: 28% ✅ (提升 3.5x)
```

### Step 4: 发现新瓶颈

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld:
  v2: 1024 conflicts

  → Bank conflict 拖累性能
```

**解决方案**: Padding ([TILE_SIZE][TILE_SIZE+1])

### Step 5: 最终验证 (v3)

```bash
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
  smsp__sass_average_data_bytes_per_sector_mem_shared,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./gemm_v3_optimized 1024 1024 1024

结果:
  - bank_conflicts: 0 ✅
  - SM Throughput: 40% ✅
  - Shared Memory efficiency: 98% ✅
```

## 💡 关键经验总结

### 1. Shared Memory 是 GEMM 优化的核心

**何时使用**:
✅ 矩阵乘法 (每个元素重用 K 次)
✅ Convolution (kernel 重用)
✅ Stencil 计算 (neighbor 重用)

❌ Element-wise 操作 (无重用)

### 2. Bank Conflict 必须处理

**简单检测**:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./kernel

如果 conflicts > 100 → 需要优化
```

**简单解决**: Padding `[N][N+1]`

### 3. Tensor Core 是性能飞跃

**投入产出比**:
- v0 → v3: 1 周开发, 40x 提升
- v3 → cuBLAS (Tensor Core): 1 行代码, 3x 提升

**建议**: 生产环境优先用 cuBLAS, 学习用手写

### 4. 优化要看 ROI

```python
# v3 已达 25% 峰值, 效率不错
# 继续优化 v3 → 可能花 1 周达到 35% (1.4x)

# 但使用 Tensor Core → 5 分钟达到 80% (6x)

ROI = (Speedup - 1) / Dev Days
v3 继续优化: (1.4 - 1) / 7 = 0.057
Tensor Core:  (6 - 1) / 0.01 = 500

→ Tensor Core ROI 高 8700 倍!
```

## 🔗 相关文档

- [CUDA/Triton 优化技术手册](../../../techniques/cuda_triton_optimization_techniques.md) - Shared Memory, Bank Conflict 详解
- [NCU UI Guide](../../../04_performance_analysis/ncu_ui_guide.md) - NCU 分析方法
- [Roofline 工具](../../../tools/roofline.py) - AI 分析
- [全局优化框架](../../../frameworks/global_optimization_framework.md) - 优化决策流程

## 📚 进阶阅读

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - Production-grade GEMM templates
- [How to Optimize GEMM](https://siboehm.com/articles/22/CUDA-MMM) - 详细 GEMM 优化博客
- [WMMA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) - Tensor Core 编程
