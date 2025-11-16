# GPU 算子性能优化完整指南

## 目录
1. [性能优化概述](#性能优化概述)
2. [内存优化](#内存优化)
3. [计算优化](#计算优化)
4. [占用率优化](#占用率优化)
5. [高级优化技巧](#高级优化技巧)
6. [实战案例](#实战案例)

---

## 性能优化概述

### 性能瓶颈分析

GPU 算子性能主要受以下因素限制：

```
性能 = min(计算吞吐量, 内存带宽)
```

#### 1. 内存受限（Memory-Bound）
- 特征：内存访问时间 >> 计算时间
- 常见算子：Element-wise 操作、Transpose、归一化
- 优化重点：减少内存访问、提高带宽利用率

#### 2. 计算受限（Compute-Bound）
- 特征：计算时间 >> 内存访问时间
- 常见算子：矩阵乘法、卷积
- 优化重点：提高计算效率、利用 Tensor Cores

#### 3. Arithmetic Intensity（算术强度）

```
Arithmetic Intensity = FLOPs / Bytes Accessed
```

- 高 AI：矩阵乘法（可以达到 100+）
- 低 AI：向量加法（~0.25）

### Roofline 模型

```
               Peak Performance (TFLOPS)
                    |
                    |     Compute Bound
                    |    /
                    |   /
                    |  /
Performance    _____|_/___________________
                   /|
                  / |  Memory Bound
                 /  |
                /   |
               /    |
              /_____|_____________________
                    |
                Arithmetic Intensity
```

**关键指标：**
- Peak Compute：GPU 的理论计算峰值（TFLOPS）
- Peak Memory BW：GPU 的理论带宽（GB/s）
- Ridge Point：AI = Peak Compute / Peak BW

---

## 内存优化

### 1. 合并访问（Coalesced Access）

#### 原理
GPU 以 32-byte、64-byte 或 128-byte 为单位访问内存（取决于架构）。

#### 好的访问模式
```cpp
// ✅ 连续访问 - 每个 warp 的线程访问连续内存
__global__ void coalesced(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // 所有线程访问连续位置
}
```

#### 差的访问模式
```cpp
// ❌ 跨步访问 - 浪费带宽
__global__ void strided(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx * stride];  // 每个线程跳跃访问
}

// ❌ 随机访问
__global__ void random(float* data, int* indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[indices[idx]];  // 随机跳转
}
```

#### 优化示例：矩阵转置

**Naive 实现（未优化）**
```cpp
__global__ void transpose_naive(float* out, const float* in, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        // 读合并，但写不合并！
        out[x * M + y] = in[y * N + x];
    }
}
```

**优化实现（使用 Shared Memory）**
```cpp
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_optimized(float* out, const float* in, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 避免 bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 合并读取
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];
        }
    }

    __syncthreads();

    // 转置后的位置
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 合并写入
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            out[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

**性能提升：** 3-5倍

### 2. Shared Memory 优化

#### Bank Conflicts

Shared Memory 被分为 32 个 bank，同一 bank 的并发访问会串行化。

```cpp
// ❌ 有 bank conflict
__shared__ float shared[32][32];
float value = shared[threadIdx.x][threadIdx.x];  // 所有线程访问同一列

// ✅ 无 bank conflict
__shared__ float shared[32][33];  // padding
float value = shared[threadIdx.x][threadIdx.y];
```

#### 使用示例：规约操作

```cpp
__global__ void reduce_optimized(float* out, const float* in, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到 shared memory
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();

    // 规约（避免 bank conflicts）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写回结果
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}
```

### 3. 数据复用

#### Tiling 技术

将大问题分解为小块（Tile），提高数据复用。

**矩阵乘法示例：**
```cpp
#define TILE_SIZE 16

__global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍历所有 tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载 tile 到 shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算部分积（数据复用 TILE_SIZE 次）
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**数据复用次数：** 每个数据被读取一次，复用 TILE_SIZE 次

### 4. 内存对齐

```cpp
// ✅ 对齐访问
float4 data = reinterpret_cast<float4*>(ptr)[idx];  // 一次读 16 bytes

// 向量化访问示例
__global__ void vectorized_copy(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < N) {
        float4 data = reinterpret_cast<const float4*>(in)[idx];
        reinterpret_cast<float4*>(out)[idx] = data;
    }
}
```

---

## 计算优化

### 1. 指令级并行（ILP）

让每个线程处理多个数据，提高 ILP。

```cpp
__global__ void ilp_optimization(float* out, const float* in, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // 每个线程处理 4 个元素
    if (idx + 3 < N) {
        float a0 = in[idx + 0];
        float a1 = in[idx + 1];
        float a2 = in[idx + 2];
        float a3 = in[idx + 3];

        // 独立的计算可以并行执行
        a0 = a0 * 2.0f + 1.0f;
        a1 = a1 * 2.0f + 1.0f;
        a2 = a2 * 2.0f + 1.0f;
        a3 = a3 * 2.0f + 1.0f;

        out[idx + 0] = a0;
        out[idx + 1] = a1;
        out[idx + 2] = a2;
        out[idx + 3] = a3;
    }
}
```

### 2. 循环展开（Loop Unrolling）

```cpp
// 使用 #pragma unroll
__global__ void unroll_example(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        if (idx * 8 + i < N) {
            data[idx * 8 + i] *= 2.0f;
        }
    }
}
```

### 3. 使用内置函数

```cpp
// ✅ 使用快速数学函数
__device__ float fast_func(float x) {
    return __expf(x);        // 快速指数
    // return __logf(x);     // 快速对数
    // return __powf(x, y);  // 快速幂
    // return __sinf(x);     // 快速正弦
}

// 使用 FMA (Fused Multiply-Add)
__device__ float fma_example(float a, float b, float c) {
    return __fmaf_rn(a, b, c);  // a * b + c (一条指令)
}
```

### 4. Tensor Cores（用于矩阵乘法）

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_gemm(
    half* C, const half* A, const half* B,
    int M, int N, int K
) {
    // WMMA (Warp Matrix Multiply-Accumulate)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 加载数据
    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, K);

    // 矩阵乘法（使用 Tensor Cores）
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 存储结果
    wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
}
```

**性能提升：** 对于 FP16，可达 10-20 倍

---

## 占用率优化

### 什么是占用率（Occupancy）？

```
Occupancy = Active Warps / Maximum Warps
```

高占用率 → 更好地隐藏延迟

### 影响因素

1. **Registers per thread**
2. **Shared memory per block**
3. **Threads per block**
4. **Blocks per SM**

### 优化策略

#### 1. 选择合适的 Block Size

```cpp
// 使用 CUDA Occupancy Calculator
int blockSize;
int minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// 或者手动调整
// 常用值：128, 256, 512, 1024
```

#### 2. 减少寄存器使用

```cpp
// 使用编译选项限制寄存器
// nvcc -maxrregcount=32 kernel.cu

// 或者在 kernel 中指定
__global__ void __launch_bounds__(256, 4)
kernel(float* data) {
    // 256 threads per block, 4 blocks per SM
}
```

#### 3. 动态 Shared Memory

```cpp
__global__ void dynamic_shared(float* data, int n) {
    extern __shared__ float shared[];
    // ...
}

// 启动时指定大小
int sharedMemSize = blockSize * sizeof(float);
kernel<<<blocks, threads, sharedMemSize>>>(data, n);
```

### 占用率 vs 性能

⚠️ **注意：** 高占用率不一定等于高性能！

```
100% occupancy ≠ Best performance
```

**关键是找到平衡点：**
- 足够的 occupancy 来隐藏延迟
- 足够的资源（registers, shared memory）来优化算法

---

## 高级优化技巧

### 1. Warp-Level 编程

#### Warp Shuffle

```cpp
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_reduce(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? in[idx] : 0.0f;

    // Warp 内规约（无需 shared memory）
    val = warp_reduce_sum(val);

    // Warp 的第一个线程写回
    if (threadIdx.x % 32 == 0) {
        atomicAdd(out, val);
    }
}
```

**优势：**
- 无需 shared memory
- 无需 `__syncthreads()`
- 更低延迟

### 2. Grid-Stride Loop

处理大于 grid size 的数据。

```cpp
__global__ void grid_stride_loop(float* data, int N) {
    // 计算 grid 的总线程数
    int gridSize = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += gridSize) {
        data[idx] *= 2.0f;
    }
}
```

**优势：**
- 适应任意大小的输入
- 提高数据复用
- 简化代码

### 3. Persistent Threads

线程持续处理多个任务。

```cpp
__global__ void persistent_kernel(float* data, int* workQueue, int numTasks) {
    __shared__ int taskId;

    while (true) {
        // 一个线程获取任务
        if (threadIdx.x == 0) {
            taskId = atomicAdd(workQueue, 1);
        }
        __syncthreads();

        if (taskId >= numTasks) break;

        // 处理任务
        int offset = taskId * blockDim.x + threadIdx.x;
        data[offset] *= 2.0f;

        __syncthreads();
    }
}
```

### 4. Stream 和异步操作

```cpp
void async_execution() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 异步拷贝和执行
    cudaMemcpyAsync(d_A1, h_A1, size, cudaMemcpyHostToDevice, stream1);
    kernel1<<<grid, block, 0, stream1>>>(d_A1, d_B1);
    cudaMemcpyAsync(h_C1, d_C1, size, cudaMemcpyDeviceToHost, stream1);

    cudaMemcpyAsync(d_A2, h_A2, size, cudaMemcpyHostToDevice, stream2);
    kernel2<<<grid, block, 0, stream2>>>(d_A2, d_B2);
    cudaMemcpyAsync(h_C2, d_C2, size, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
}
```

### 5. Cooperative Groups

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cg_kernel(float* data) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    // Tile 内规约
    float val = data[threadIdx.x];
    val = cg::reduce(tile32, val, cg::plus<float>());

    if (tile32.thread_rank() == 0) {
        data[threadIdx.x / 32] = val;
    }
}
```

---

## 实战案例

### 案例 1：Softmax 优化

#### Naive 实现
```cpp
__global__ void softmax_naive(float* out, const float* in, int N) {
    int idx = blockIdx.x;

    // Find max
    float max_val = -INFINITY;
    for (int i = 0; i < N; i++) {
        max_val = fmaxf(max_val, in[idx * N + i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += expf(in[idx * N + i] - max_val);
    }

    // Normalize
    for (int i = 0; i < N; i++) {
        out[idx * N + i] = expf(in[idx * N + i] - max_val) / sum;
    }
}
```

**问题：** 多次遍历、串行执行

#### 优化实现
```cpp
__global__ void softmax_optimized(float* out, const float* in, int N) {
    __shared__ float shared_max;
    __shared__ float shared_sum;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Parallel max reduction
    float thread_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        thread_max = fmaxf(thread_max, in[row * N + i]);
    }

    // Warp-level reduction
    thread_max = warp_reduce_max(thread_max);

    if (tid % 32 == 0) {
        atomicMax(&shared_max, thread_max);  // 需要 float atomicMax
    }
    __syncthreads();

    // Parallel exp and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        thread_sum += expf(in[row * N + i] - shared_max);
    }

    thread_sum = warp_reduce_sum(thread_sum);

    if (tid % 32 == 0) {
        atomicAdd(&shared_sum, thread_sum);
    }
    __syncthreads();

    // Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        out[row * N + i] = expf(in[row * N + i] - shared_max) / shared_sum;
    }
}
```

**优化点：**
- 并行规约
- Warp shuffle
- 减少全局内存访问

**性能提升：** 5-10 倍

### 案例 2：LayerNorm 优化

```cpp
__global__ void layernorm_optimized(
    float* out, const float* in,
    const float* gamma, const float* beta,
    int N, float eps
) {
    __shared__ float s_mean, s_var;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Compute mean (parallel)
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += in[row * N + i];
    }
    sum = warp_reduce_sum(sum);

    if (tid == 0) {
        s_mean = sum / N;
    }
    __syncthreads();

    // Compute variance (parallel)
    float var = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = in[row * N + i] - s_mean;
        var += diff * diff;
    }
    var = warp_reduce_sum(var);

    if (tid == 0) {
        s_var = var / N;
    }
    __syncthreads();

    // Normalize (parallel)
    float inv_std = rsqrtf(s_var + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        float norm = (in[row * N + i] - s_mean) * inv_std;
        out[row * N + i] = gamma[i] * norm + beta[i];
    }
}
```

---

## 性能检查清单

### 内存优化 ✓
- [ ] 使用 coalesced 访问模式
- [ ] 利用 shared memory 减少全局内存访问
- [ ] 避免 bank conflicts
- [ ] 使用向量化访问（float4, etc.）
- [ ] 数据复用（tiling）

### 计算优化 ✓
- [ ] 使用快速数学函数
- [ ] 循环展开
- [ ] 提高 ILP
- [ ] 使用 Tensor Cores（如果适用）
- [ ] 减少分支 divergence

### 占用率优化 ✓
- [ ] 选择合适的 block size
- [ ] 平衡寄存器使用
- [ ] 优化 shared memory 使用
- [ ] 检查占用率（使用 `--ptxas-options=-v`）

### 高级优化 ✓
- [ ] 使用 warp-level 原语
- [ ] 异步操作和 streams
- [ ] Grid-stride loops
- [ ] Cooperative groups

---

## 性能测试模板

```cpp
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

float benchmark_kernel(void (*kernel)(float*, int), float* d_data, int N, int iterations = 100) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel(d_data, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel(d_data, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds / iterations;
}

void print_bandwidth(float ms, size_t bytes) {
    float bandwidth = (bytes / 1e9) / (ms / 1000);  // GB/s
    printf("Time: %.3f ms, Bandwidth: %.2f GB/s\n", ms, bandwidth);
}
```

---

## 下一步

- 学习 [GPU 架构适配](gpu_architecture.md)
- 查看 [性能分析工具](../04_performance_analysis/ncu_guide.md)
- 阅读 [最佳实践](best_practices.md)
