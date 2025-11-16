# CUDA/Triton 优化技术完全手册

> **目标**：详细讲解每个优化技术的原理、使用场景、CUDA/Triton 实现、NCU 验证方法

---

## 目录

1. [Memory Coalescing（内存合并访问）](#1-memory-coalescing内存合并访问)
2. [Vectorization（向量化）](#2-vectorization向量化)
3. [Shared Memory](#3-shared-memory)
4. [Kernel Fusion（算子融合）](#4-kernel-fusion算子融合)
5. [ILP（指令级并行）](#5-ilp指令级并行)
6. [Loop Unrolling（循环展开）](#6-loop-unrolling循环展开)
7. [Tensor Core](#7-tensor-core)
8. [Bank Conflict 避免](#8-bank-conflict-避免)
9. [Warp Divergence 优化](#9-warp-divergence-优化)
10. [Occupancy 优化](#10-occupancy-优化)

---

## 1. Memory Coalescing（内存合并访问）

### 原理

**硬件机制**：
- GPU 以 **128-byte** 为单位访问内存（cache line）
- 一个 warp (32 threads) 发出一次内存请求
- 如果 32 threads 访问的地址连续，可以合并成 **1 次**内存事务
- 如果地址分散，需要多次内存事务 → 浪费带宽

**理想情况**：
```
Thread 0: data[0]
Thread 1: data[1]
...
Thread 31: data[31]

→ 32 个 float (128 bytes) 刚好 1 个 cache line
→ 1 次内存事务 ✓
```

**糟糕情况**：
```
Thread 0: data[0]
Thread 1: data[64]    ← 跨度大
Thread 2: data[128]
...

→ 每个 thread 访问不同的 cache line
→ 32 次内存事务 ✗
→ 带宽浪费 32x
```

### 何时使用

**NCU 指标判断**：

```
打开 NCU-UI → Details → Memory Workload Analysis → 查看：
l1tex__average_t_sectors_per_request

判断标准：
- 1.0-1.2：完美 coalesced ✓
- 1.2-2.0：良好
- 2.0-4.0：一般，有优化空间
- > 4.0：很差，必须优化

带宽浪费 = (ratio - 1) / ratio × 100%
预期加速比 = ratio
```

**触发条件**：
- Sectors Per Request > 1.5
- Memory Throughput 高但性能差
- Memory-bound kernel

### CUDA 实现

#### ❌ 错误：跨步访问

```c
__global__ void strided_access(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = 64;  // 大步长

    if (tid * stride < N) {
        float val = data[tid * stride];  // 每个thread相距64个元素
        // ... 处理
    }
}

/*
NCU 结果：
l1tex__average_t_sectors_per_request: 64.0
带宽浪费：98.4%
*/
```

#### ✅ 正确：连续访问

```c
__global__ void coalesced_access(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float val = data[tid];  // 连续访问
        // ... 处理
    }
}

/*
NCU 结果：
l1tex__average_t_sectors_per_request: 1.0
带宽效率：100%
加速比：64x
*/
```

#### 实战案例：Matrix Transpose

```c
// ❌ 错误：写入时不连续
__global__ void transpose_naive(float* in, float* out, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        // 读取：连续 ✓
        float val = in[y * N + x];

        // 写入：跨步访问 ✗
        out[x * N + y] = val;  // 相邻thread写入相距N个元素
    }
}

// ✅ 正确：使用 Shared Memory
__global__ void transpose_shared(float* in, float* out, int N) {
    __shared__ float tile[32][32+1];  // +1 避免bank conflict

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // 读取到shared memory（连续）
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    // 从shared memory写出（转置后连续）
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < N && y < N) {
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/*
性能对比：
transpose_naive:  l1tex__average_t_sectors_per_request = 32.0
                  Time: 5.2 ms

transpose_shared: l1tex__average_t_sectors_per_request = 1.0
                  Time: 0.8 ms

加速比：6.5x
*/
```

### Triton 实现

```python
import triton
import triton.language as tl

# Triton 自动处理 coalescing！
@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Triton 自动确保 offsets 是连续的
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 自动 coalesced 访问
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# 注意：Triton 会自动优化访问模式
# 但如果你手动计算不连续的 offsets，仍然会有问题

# ❌ 错误：手动创建跨步访问
@triton.jit
def bad_kernel(data_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE * 64 + tl.arange(0, BLOCK_SIZE) * 64  # stride=64
    # ... 这样仍然会不 coalesced
```

### NCU 验证方法

```bash
# 优化前
ncu --metrics l1tex__average_t_sectors_per_request -o before ./program

# 优化后
ncu --metrics l1tex__average_t_sectors_per_request -o after ./program

# 对比
ncu-ui before.ncu-rep after.ncu-rep
```

**在 NCU-UI 中查看**：
```
Details → Memory Workload Analysis → Tables

Before:
l1tex__average_t_sectors_per_request: 8.5

After:
l1tex__average_t_sectors_per_request: 1.1 ✓

加速比验证：Summary → Duration
Before: 25 ms
After: 3 ms
实际加速：8.3x ≈ 理论 8.5x ✓
```

### 注意事项

1. **不仅读取，写入也要 coalesced**
   - 读和写都会检查
   - 有时读取 coalesced，但写入不是

2. **对齐很重要**
   - 起始地址最好是 128 字节对齐
   - `cudaMalloc` 自动对齐，但手动偏移要小心

3. **Shared Memory 可以解决写入不连续的问题**
   - 见 transpose 例子

4. **结构体访问**
   ```c
   // ❌ AoS (Array of Structures) - 不连续
   struct Point { float x, y, z; };
   Point points[N];
   float x = points[tid].x;  // 相邻thread相距sizeof(Point)

   // ✅ SoA (Structure of Arrays) - 连续
   float x[N], y[N], z[N];
   float val = x[tid];  // 连续 ✓
   ```

---

## 2. Vectorization（向量化）

### 原理

**硬件机制**：
- 现代 GPU 支持向量化 load/store
- `float4` 一次读取 128 bits（4 个 float）
- 一条指令完成 4 个元素的操作
- 减少指令数量，提高带宽利用率

**为什么有效**：
```
标量访问：
load r0, [addr]       // 32 bits
load r1, [addr+4]     // 32 bits
load r2, [addr+8]     // 32 bits
load r3, [addr+12]    // 32 bits
→ 4 条指令

向量化访问：
load.v4 {r0,r1,r2,r3}, [addr]  // 128 bits
→ 1 条指令，4x 效率
```

### 何时使用

**NCU 指标判断**：

```
前提：
1. 已经 coalesced（Sectors Per Request ≈ 1.0）
2. Memory-bound
3. DRAM Throughput < 90%

判断：
如果以上都满足，且访问是 4 字节对齐
→ 使用 vectorization
→ 预期：1.5-2x 加速
```

**触发条件**：
- Element-wise 操作
- 数据类型是 float 或 int32
- 访问模式已经连续
- 带宽利用率还未饱和

### CUDA 实现

#### 基础：float4

```c
// ❌ 标量版本
__global__ void vector_add_scalar(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

// ✅ 向量化版本
__global__ void vector_add_vectorized(float4* a, float4* b, float4* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N / 4) {
        float4 va = a[tid];  // 一次读取4个float
        float4 vb = b[tid];

        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        c[tid] = vc;  // 一次写入4个float
    }
}

// 调用
int N = 1024 * 1024;
float* d_a, *d_b, *d_c;
cudaMalloc(&d_a, N * sizeof(float));
// ...

// 标量版本
dim3 block(256);
dim3 grid((N + 255) / 256);
vector_add_scalar<<<grid, block>>>(d_a, d_b, d_c, N);

// 向量化版本
dim3 grid_vec((N/4 + 255) / 256);
vector_add_vectorized<<<grid_vec, block>>>(
    (float4*)d_a, (float4*)d_b, (float4*)d_c, N
);

/*
NCU 结果：
标量版本：
  DRAM Throughput: 65%
  Duration: 2.5 ms

向量化版本：
  DRAM Throughput: 85%
  Duration: 1.5 ms

加速比：1.67x
*/
```

#### 高级：处理非4倍数长度

```c
__global__ void vector_add_hybrid(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 主体：向量化处理
    int vec_tid = tid;
    int vec_N = N / 4;

    if (vec_tid < vec_N) {
        float4 va = ((float4*)a)[vec_tid];
        float4 vb = ((float4*)b)[vec_tid];

        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        ((float4*)c)[vec_tid] = vc;
    }

    // 尾部：标量处理剩余元素
    int remainder_start = vec_N * 4;
    int scalar_tid = remainder_start + tid;

    if (scalar_tid < N) {
        c[scalar_tid] = a[scalar_tid] + b[scalar_tid];
    }
}
```

#### 其他向量类型

```c
// int4, int2, float2 等
__global__ void example_types(void* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // float2 (64 bits)
    float2 v2 = ((float2*)data)[tid];

    // double2 (128 bits)
    double2 d2 = ((double2*)data)[tid];

    // char4 (32 bits)
    char4 c4 = ((char4*)data)[tid];
}
```

### Triton 实现

```python
import triton
import triton.language as tl

# Triton 自动向量化！
@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # Triton 编译器会自动向量化这些操作
    # 如果 BLOCK_SIZE >= 4，会生成 vectorized load/store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# 使用
def vector_add(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()

    # BLOCK_SIZE 越大，向量化程度越高
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=1024,  # Triton 会自动优化
    )
    return output

# Triton 的优势：
# 1. 自动向量化
# 2. 自动选择最优向量宽度
# 3. 不需要手动处理边界
```

### NCU 验证方法

```bash
ncu --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    smsp__inst_executed.sum \
    -o vectorized ./program
```

**在 NCU-UI 中查看**：
```
Details → Compute Workload Analysis

标量版本：
Instructions Executed: 1,000,000

向量化版本：
Instructions Executed: 250,000  ← 减少了 4x ✓

同时检查：
Memory Workload Analysis → DRAM Throughput
应该从 65% 提升到 85%+
```

### 注意事项

1. **对齐要求**
   ```c
   // 数据必须对齐到向量大小
   float* data;
   cudaMalloc(&data, N * sizeof(float));  // 自动对齐 ✓

   // 如果手动偏移，确保对齐
   float4* vec_data = (float4*)(data + offset);
   // offset 必须是 4 的倍数
   ```

2. **不是所有操作都能向量化**
   ```c
   // ✅ 可以向量化
   c = a + b;
   c = a * b;
   c = fmaxf(a, b);

   // ❌ 不能直接向量化
   if (a > b) c = a;  // 分支
   c = __expf(a);     // 特殊函数（部分支持）
   ```

3. **Triton 更智能**
   - Triton 会自动分析是否可以向量化
   - 会选择最优的向量宽度（可能不是 float4）
   - 建议：对于复杂操作，优先用 Triton

4. **性能提升有限**
   - 如果已经是 memory-bound 且带宽饱和，提升有限
   - 主要收益：减少指令数，提高带宽利用率
   - 预期：1.5-2x，不会更高

---

## 3. Shared Memory

### 原理

**内存层次**：
```
Registers:      ~1 cycle,   64KB/SM,    私有
Shared Memory:  ~5 cycles,  96KB/SM,    Block 内共享
L1 Cache:       ~30 cycles, 192KB/SM,   自动管理
L2 Cache:       ~200 cycles, 40MB,      全局共享
HBM (DRAM):     ~400 cycles, 40GB,      全局
```

**Shared Memory 的优势**：
- **速度快**：比 Global Memory 快 100x
- **带宽高**：~20 TB/s vs 1.5 TB/s (HBM)
- **可控**：程序员显式管理

**使用场景**：数据被 Block 内多个 threads **复用**

### 何时使用

**NCU 指标判断**：

```
打开 NCU-UI → Details → Memory Workload Analysis

判断条件：
1. L2 Cache Hit Rate < 50%  ← 数据没有复用
2. 同一数据被多次访问（代码层面判断）
3. 数据可以放入 Shared Memory（< 96KB per block）

预期收益：
- 如果数据复用 10 次：理论加速 ~5-10x
- 实际：2-5x（考虑同步开销）
```

**典型场景**：
- Matrix Multiplication（Tiling）
- Convolution
- Reduction
- Scan/Prefix Sum
- Transpose

### CUDA 实现

#### 案例 1：Matrix Multiplication (GEMM)

```c
#define TILE_SIZE 32

// ❌ Naive GEMM（无 Shared Memory）
__global__ void gemm_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // 每次循环都从 Global Memory 读取
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/*
问题分析：
- A[row * N + k] 被读取 N 次（k = 0..N-1）
- B[k * N + col] 被读取 N 次
- L2 Cache 无法缓存所有数据（Working Set 太大）
- 每个元素平均从 HBM 读取 N/2 次

NCU 结果：
L2 Hit Rate: 15%
Time: 500 ms (N=4096)
*/

// ✅ Tiled GEMM（使用 Shared Memory）
__global__ void gemm_shared(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Tiling：分块计算
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 协作加载：每个 thread 加载一个元素到 Shared Memory
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t * TILE_SIZE + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // 确保所有数据都加载完成

        // 计算：从 Shared Memory 读取（快！）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // 确保所有 thread 都计算完成再进入下一轮
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

/*
性能分析：
- 每个 tile 的数据从 Global Memory 读取 1 次
- 在 Shared Memory 中复用 TILE_SIZE 次
- 数据复用率：32x

NCU 结果：
L2 Hit Rate: 仍然 ~15%（Working Set 大）
但 Shared Memory 命中率: ~95%
Time: 50 ms (N=4096)
加速比：10x
*/
```

#### 案例 2：Reduction

```c
// ❌ Naive Reduction（无 Shared Memory）
__global__ void reduce_naive(float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // 每个 thread 将结果写回 Global Memory
        atomicAdd(output, input[tid]);  // 非常慢！
    }
}

// ✅ Shared Memory Reduction
__global__ void reduce_shared(float* input, float* output, int N) {
    __shared__ float sdata[256];  // 假设 blockDim.x = 256

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个 thread 加载数据到 Shared Memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in Shared Memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 只有 thread 0 写回 Global Memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/*
性能对比：
naive:  大量 atomicAdd 到 Global Memory
        Time: 10 ms

shared: 只有每个 block 的最终结果写回
        Time: 0.5 ms

加速比：20x
*/
```

### Triton 实现

```python
import triton
import triton.language as tl

# Triton 中的 Shared Memory 使用起来更简单
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 累加器（在寄存器中）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiling
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # 加载 A tile（Triton 自动使用 Shared Memory！）
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

        # 加载 B tile
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))

        # 矩阵乘法（Triton 自动优化）
        acc += tl.dot(a, b)

    # 写回结果
    c = acc.to(tl.float16)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 注意：Triton 的优势
# 1. 自动管理 Shared Memory
# 2. 自动插入 __syncthreads()
# 3. 自动优化数据布局
# 4. 不需要手动计算 tile 索引
```

### NCU 验证方法

```bash
ncu --metrics \
    l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_global.sum \
    -o shared ./program
```

**在 NCU-UI 中查看**：
```
Details → Memory Workload Analysis

优化前（无 Shared Memory）：
Global Memory Accesses: 10,000,000
Shared Memory Accesses: 0
L2 Hit Rate: 15%

优化后（有 Shared Memory）：
Global Memory Accesses: 1,000,000  ← 减少 10x
Shared Memory Accesses: 9,000,000  ← 大部分访问在 Shared Memory
L2 Hit Rate: 仍然 ~15%（但无所谓，因为用了 Shared Memory）

Time: 500 ms → 50 ms
加速比：10x ✓
```

### 注意事项

1. **Shared Memory 大小限制**
   ```c
   // A100: 每个 Block 最多 96KB Shared Memory（动态配置）
   // 如果超过，Occupancy 会下降

   // 查看使用量
   __shared__ float data[1024];  // 4KB
   __shared__ float data2[10000]; // 40KB
   // 总共 44KB < 96KB ✓

   // 如果超过 96KB，需要：
   // 1. 减小 tile size
   // 2. 或接受更低的 Occupancy
   ```

2. **Bank Conflicts**（见后面章节）
   - Shared Memory 分为 32 个 bank
   - 如果多个 thread 访问同一个 bank → conflict → 性能下降
   - 需要特别注意访问模式

3. **同步开销**
   ```c
   __syncthreads();  // 有开销

   // 如果 __syncthreads() 太多，检查：
   NCU → Warp State Statistics
   stall_barrier: 如果 > 20%，需要优化
   ```

4. **何时不用 Shared Memory**
   - Element-wise 操作（无复用）
   - 数据只访问一次
   - Working Set 很小，L1/L2 Cache 够用

---

## 4. Kernel Fusion（算子融合）

### 原理

**问题**：多个小 kernel 的开销
```
Kernel 1: ReLU       → 读 x, 写 x  (kernel launch ~5μs)
Kernel 2: Add Bias   → 读 x, 写 x  (kernel launch ~5μs)
Kernel 3: Scale      → 读 x, 写 x  (kernel launch ~5μs)

总开销：
- 3 次 kernel launch: ~15-30 μs
- 3 次 Global Memory 读写: 每次从 HBM 读写一次
- 中间结果存储：占用内存
```

**融合后**：
```
Kernel Fused: ReLU + Add + Scale
  → 读 x 一次, 写 result 一次
  → 1 次 kernel launch
  → 中间结果在寄存器中
```

**收益**：
- 减少 kernel launch overhead
- 减少 Global Memory 访问
- 提高 L2 Cache hit rate（数据局部性更好）
- 减少内存占用

### 何时使用

**NCU 指标判断**：

```
使用 nsys（不是 ncu）先看全局：

nsys profile -o timeline ./program
nsys-ui timeline.qdrep

查看：
1. 有多个连续的小 kernel（< 100 μs each）
2. kernel 之间有数据依赖

NCU 指标：
L2 Cache Hit Rate < 50%  ← 数据没有跨 kernel 复用

预期收益：
- 2-3 个 kernel 融合：2-5x
- 5+ 个 kernel 融合：3-10x
```

**触发条件**：
- 多个 element-wise 操作
- Kernel duration < 10 ms each
- 存在 producer-consumer 关系

### CUDA 实现

#### 案例：Fused Activation

```c
// ❌ 未融合：3 个 kernel
__global__ void relu(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] = fmaxf(0.0f, data[tid]);
    }
}

__global__ void add_bias(float* data, float bias, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] += bias;
    }
}

__global__ void scale(float* data, float alpha, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        data[tid] *= alpha;
    }
}

// 调用
relu<<<grid, block>>>(data, N);
add_bias<<<grid, block>>>(data, 0.1f, N);
scale<<<grid, block>>>(data, 2.0f, N);

/*
性能：
- Kernel 1: 5 ms
- Kernel 2: 5 ms
- Kernel 3: 5 ms
- Launch overhead: 3 × 7μs = 21μs
- Total: 15.021 ms

Memory Access:
- 3 次读 data (N × 4 bytes × 3 = 12N bytes)
- 3 次写 data (12N bytes)
- Total: 24N bytes
*/

// ✅ 融合：1 个 kernel
__global__ void fused_activation(float* data, float bias, float alpha, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        float val = data[tid];      // 读一次
        val = fmaxf(0.0f, val);     // ReLU（寄存器）
        val += bias;                 // Add（寄存器）
        val *= alpha;                // Scale（寄存器）
        data[tid] = val;            // 写一次
    }
}

// 调用
fused_activation<<<grid, block>>>(data, 0.1f, 2.0f, N);

/*
性能：
- Kernel: 3 ms
- Launch overhead: 7μs
- Total: 3.007 ms

加速比：15.021 / 3.007 = 5x

Memory Access:
- 1 次读 data (4N bytes)
- 1 次写 data (4N bytes)
- Total: 8N bytes（减少了 3x）

NCU 验证：
L2 Hit Rate: 优化前 20% → 优化后 80%（数据在一个 kernel 内复用）
```

#### 案例：LayerNorm + Linear (更复杂)

```c
// ❌ 未融合
__global__ void layer_norm(float* x, float* out, float* gamma, float* beta, int N) {
    // LayerNorm 实现
    // ...
}

__global__ void linear(float* x, float* weight, float* bias, float* out, int M, int N) {
    // Linear 实现
    // ...
}

// 调用
layer_norm<<<grid1, block>>>(x, ln_out, gamma, beta, N);
linear<<<grid2, block>>>(ln_out, weight, bias, final_out, M, N);

// ✅ 融合（复杂，但有效）
__global__ void fused_ln_linear(
    float* x,           // [N]
    float* gamma,       // [N]
    float* beta,        // [N]
    float* weight,      // [N, M]
    float* bias,        // [M]
    float* out,         // [M]
    int N, int M
) {
    __shared__ float mean, var;
    __shared__ float s_x[256];  // 假设 N <= 256

    int tid = threadIdx.x;

    // Step 1: Load input to shared memory
    if (tid < N) {
        s_x[tid] = x[tid];
    }
    __syncthreads();

    // Step 2: Compute mean
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += s_x[i];
        }
        mean = sum / N;
    }
    __syncthreads();

    // Step 3: Compute variance
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = s_x[i] - mean;
            sum_sq += diff * diff;
        }
        var = sum_sq / N;
    }
    __syncthreads();

    // Step 4: Normalize (in shared memory)
    if (tid < N) {
        s_x[tid] = (s_x[tid] - mean) * rsqrtf(var + 1e-5f) * gamma[tid] + beta[tid];
    }
    __syncthreads();

    // Step 5: Linear (直接用 shared memory 的数据)
    for (int i = tid; i < M; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += s_x[j] * weight[j * M + i];
        }
        out[i] = sum + bias[i];
    }
}

/*
收益：
- 避免 LayerNorm 的输出写回 Global Memory
- Linear 直接从 Shared Memory 读取
- 减少一次 kernel launch

性能：
未融合: 5 ms (LN) + 10 ms (Linear) + overhead = 15 ms
融合:   8 ms
加速比: 1.875x
```

### Triton 实现

```python
import triton
import triton.language as tl

# Triton 让 fusion 变得非常简单！
@triton.jit
def fused_activation_kernel(
    x_ptr, output_ptr,
    bias, alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 一次性完成所有操作
    x = tl.load(x_ptr + offsets, mask=mask)

    # ReLU
    x = tl.where(x > 0, x, 0.0)

    # Add bias
    x = x + bias

    # Scale
    x = x * alpha

    # 一次写回
    tl.store(output_ptr + offsets, x, mask=mask)

# PyTorch 2.0 torch.compile 也支持自动融合
import torch

@torch.compile
def fused_ops(x, bias, alpha):
    x = torch.relu(x)
    x = x + bias
    x = x * alpha
    return x

# PyTorch 会自动融合这些操作！
```

### NCU 验证方法

```bash
# 需要用 nsys 看全局
nsys profile -o before python train.py

# 融合后
nsys profile -o after python train.py

# 对比
nsys-ui before.qdrep after.qdrep
```

**在 nsys-ui 中查看**：
```
Timeline:
Before:
  Kernel1 ─┬─ 5ms
           │ gap (~10μs)
           └─ Kernel2 ─┬─ 5ms
                       │ gap
                       └─ Kernel3 ─── 5ms

After:
  KernelFused ───────── 3ms

加速比：15 / 3 = 5x ✓
```

### 注意事项

1. **不是所有 kernel 都应该融合**
   ```
   ❌ 不融合：
   - Kernel A: GEMM (compute-bound, 100 ms)
   - Kernel B: Element-wise Add (memory-bound, 1 ms)

   原因：
   - GEMM 已经很大，融合后更复杂
   - Add 只占 1%，融合收益低
   - 可能无法使用 cuBLAS/Tensor Core

   ✅ 融合：
   - 多个 element-wise 操作
   - 都是 memory-bound
   - 每个都很小（< 10 ms）
   ```

2. **寄存器压力**
   ```c
   // 融合太多操作可能导致寄存器溢出
   __global__ void overly_fused(...) {
       float a, b, c, d, e, f, g, h;  // 太多寄存器
       // ...
   }

   // 检查：
   NCU → Occupancy → Limiting Factor
   如果是 "Registers" → 融合过度，需要拆分
   ```

3. **正确性**
   ```c
   // 确保融合后逻辑正确
   // 特别注意数据依赖

   // ❌ 错误融合（有依赖）
   Kernel A: x[i] = x[i-1] + 1  // 依赖前一个元素
   Kernel B: y[i] = x[i] * 2
   // 不能融合！Kernel A 必须全部完成后才能执行 B

   // ✅ 可以融合（无依赖）
   Kernel A: x[i] = a[i] + b[i]
   Kernel B: y[i] = x[i] * 2
   // 可以融合成: y[i] = (a[i] + b[i]) * 2
   ```

4. **框架支持**
   - PyTorch 2.0 `torch.compile` 会自动融合
   - TensorRT 有 fusion 优化
   - Triton 非常适合手动 fusion
   - 纯 CUDA：需要手动实现，但完全可控

---

## 5. ILP（指令级并行）

### 原理

**CPU vs GPU ILP**：
```
CPU: 乱序执行，硬件自动提取 ILP
GPU: 顺序执行，需要程序员显式提供 ILP
```

**依赖链问题**：
```c
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += data[i];  // 每次迭代依赖上一次的 sum
}

// 指令流：
load  r1, data[0]
fadd  sum, sum, r1   ← 依赖 sum
load  r2, data[1]
fadd  sum, sum, r2   ← 依赖上一个 fadd 的结果
...

// 问题：每个 fadd 必须等待前一个完成
// → 无法并行执行
// → Issue Efficiency 低
```

**增加 ILP**：
```c
float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
for (int i = 0; i < N; i += 4) {
    sum0 += data[i+0];  // 4 个独立的依赖链
    sum1 += data[i+1];  // 可以并行执行
    sum2 += data[i+2];
    sum3 += data[i+3];
}
float sum = sum0 + sum1 + sum2 + sum3;

// 指令可以并行发射
// → Issue Efficiency 高
```

### 何时使用

**NCU 指标判断**：

```
打开 NCU-UI → Details → Compute Workload Analysis

关键指标：
1. smsp__issue_active.pct (Issue Efficiency)
   - < 60%：ILP 不足，可以优化
   - > 80%：ILP 已经足够

2. smsp__warp_issue_stalled_short_scoreboard.pct
   - > 20%：等待算术运算结果，ILP 不足

判断：
SM Throughput > 60% (有一定计算)
但 Issue Efficiency < 60%
→ 使用 ILP 优化
→ 预期：1.3-2x
```

**触发条件**：
- Compute-bound kernel
- 有循环累加
- Issue Efficiency < 60%

### CUDA 实现

#### 案例 1：Reduction

```c
// ❌ 低 ILP（单个累加器）
__global__ void reduce_low_ilp(float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;  // 单个累加器

    // 每个 thread 累加多个元素
    for (int j = i; j < N; j += blockDim.x * gridDim.x) {
        sum += input[j];  // 依赖链
    }

    sdata[tid] = sum;
    __syncthreads();

    // ... reduction in shared memory
}

/*
NCU 结果：
Issue Efficiency: 45%
Short Scoreboard Stall: 30%
Time: 2.0 ms
*/

// ✅ 高 ILP（多个累加器）
__global__ void reduce_high_ilp(float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;  // ×4

    // 4 个独立累加器
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    // 每次循环处理 4 个元素
    for (int j = i; j < N; j += blockDim.x * gridDim.x * 4) {
        if (j < N) sum0 += input[j];
        if (j + blockDim.x * gridDim.x < N) sum1 += input[j + blockDim.x * gridDim.x];
        if (j + blockDim.x * gridDim.x * 2 < N) sum2 += input[j + blockDim.x * gridDim.x * 2];
        if (j + blockDim.x * gridDim.x * 3 < N) sum3 += input[j + blockDim.x * gridDim.x * 3];
    }

    // 合并累加器
    float sum = sum0 + sum1 + sum2 + sum3;

    sdata[tid] = sum;
    __syncthreads();

    // ... reduction in shared memory
}

/*
NCU 结果：
Issue Efficiency: 75%  ← 提升
Short Scoreboard Stall: 10%  ← 减少
Time: 1.2 ms

加速比：1.67x
*/
```

#### 案例 2：GEMM (寄存器 Blocking)

```c
// ❌ 低 ILP
__global__ void gemm_low_ilp(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;  // 单个累加器
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];  // 依赖链
        }
        C[row * N + col] = sum;
    }
}

// ✅ 高 ILP（寄存器 Blocking: 每个 thread 计算 4x4 个输出）
__global__ void gemm_high_ilp(float* A, float* B, float* C, int N) {
    int base_row = blockIdx.y * blockDim.y * 4 + threadIdx.y * 4;
    int base_col = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    // 16 个独立累加器（4x4）
    float sum[4][4] = {0.0f};

    for (int k = 0; k < N; k++) {
        // 加载 A 和 B 的数据
        float a[4], b[4];

        for (int i = 0; i < 4; i++) {
            a[i] = A[(base_row + i) * N + k];
            b[i] = B[k * N + (base_col + i)];
        }

        // 16 个独立的乘加操作
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                sum[i][j] += a[i] * b[j];  // 独立！
            }
        }
    }

    // 写回结果
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (base_row + i < N && base_col + j < N) {
                C[(base_row + i) * N + (base_col + j)] = sum[i][j];
            }
        }
    }
}

/*
NCU 对比：
Issue Efficiency: 50% → 80%
Time: 100 ms → 60 ms
加速比：1.67x
*/
```

### Triton 实现

```python
# Triton 编译器会自动优化 ILP
# 但可以通过 blocking 显式提高

@triton.jit
def matmul_high_ilp(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 累加器（BLOCK_M × BLOCK_N 个独立累加器）
    # Triton 会自动分配到寄存器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # 加载数据
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])

        # 矩阵乘法（Triton 自动展开，提供高 ILP）
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)

# 关键：BLOCK_M × BLOCK_N 决定了 ILP
# 更大的 block → 更多独立累加器 → 更高 ILP
# 但也受限于寄存器数量
```

### NCU 验证方法

```bash
ncu --metrics \
    smsp__issue_active.pct,\
    smsp__warp_issue_stalled_short_scoreboard.pct,\
    smsp__average_warps_issue_stalled_short_scoreboard.ratio \
    -o ilp ./program
```

**在 NCU-UI 中查看**：
```
Details → Compute Workload Analysis

优化前：
Issue Efficiency (smsp__issue_active): 45%
Short Scoreboard Stall: 30%

优化后：
Issue Efficiency: 75%  ← 提升
Short Scoreboard Stall: 10%  ← 减少

Warp State Statistics:
stall_short_scoreboard: 30% → 10% ✓
```

### 注意事项

1. **寄存器限制**
   ```c
   // 过多的累加器会导致寄存器溢出
   float sum[100];  // 可能超过寄存器限制

   // NCU 检查：
   Occupancy → Limiting Factor
   如果是 "Registers" → 减少累加器数量

   // 经验：
   // 4-8 个累加器通常是最佳的
   // 16 个对于 GEMM 合适
   ```

2. **ILP 不是万能的**
   ```
   前提：Compute-bound

   如果是 Memory-bound:
   → 提高 ILP 无效
   → 应该优化内存访问
   ```

3. **与向量化的区别**
   ```
   向量化: 使用 SIMD 指令（float4）
   ILP: 提供多个独立操作让硬件调度

   两者可以结合使用！
   ```

4. **自动优化**
   ```
   // 编译器优化
   nvcc -O3 --use_fast_math  // 会自动展开一些循环

   // Triton 更激进的自动优化
   ```

---

## 6. Loop Unrolling（循环展开）

### 原理

**编译器优化**：
- 减少循环控制开销（条件判断、跳转）
- 增加指令级并行(ILP)
- 减少分支预测失败

**示例**：
```c
// 未展开
for (int i = 0; i < 4; i++) {
    sum += data[i];
}
// → 4 次循环，4 次条件判断，4 次跳转

// 展开
sum += data[0];
sum += data[1];
sum += data[2];
sum += data[3];
// → 0 次循环，0 次条件判断
// → 且4个加法可以并行（如果是独立累加器）
```

### 何时使用

**NCU 指标判断**：

```
场景：
1. 小循环（迭代次数少，如 < 16）
2. Compute-bound
3. Issue Efficiency < 70%

判断：
与 ILP 类似，检查 Issue Efficiency

预期收益：
- 小循环（< 8 iter）：1.2-1.5x
- 配合 ILP：1.5-2x
```

**触发条件**：
- 循环次数已知且小
- 循环体简单
- 配合 ILP 使用

### CUDA 实现

#### 手动展开

```c
// ❌ 未展开
__global__ void matmul_no_unroll(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {  // 大循环，不展开
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ✅ 部分展开（Tiling 内部展开）
#define TILE_SIZE 32
__global__ void matmul_unroll(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 加载到 shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // 内层循环展开（这是关键！）
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {  // 编译器会展开这个
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// 或者完全手动展开（小循环）
__global__ void vector_dot_unroll(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N / 8) {
        int base = tid * 8;

        // 手动展开 8 次
        float sum = 0.0f;
        sum += a[base+0] * b[base+0];
        sum += a[base+1] * b[base+1];
        sum += a[base+2] * b[base+2];
        sum += a[base+3] * b[base+3];
        sum += a[base+4] * b[base+4];
        sum += a[base+5] * b[base+5];
        sum += a[base+6] * b[base+6];
        sum += a[base+7] * b[base+7];

        c[tid] = sum;
    }
}
```

#### Pragma 指令

```c
// #pragma unroll：让编译器自动展开
#pragma unroll
for (int i = 0; i < 4; i++) {
    sum += data[i];
}

// #pragma unroll N：展开 N 次
#pragma unroll 8
for (int i = 0; i < 32; i++) {  // 展开 8 次，剩余部分仍是循环
    sum += data[i];
}

// #pragma unroll 1：不展开
#pragma unroll 1
for (int i = 0; i < N; i++) {
    // ...
}
```

### Triton 实现

```python
# Triton 编译器会自动展开循环
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ...

    for k in range(0, K, BLOCK_K):  # Triton 会智能展开
        # ...
        acc += tl.dot(a, b)

# Triton 的优势：
# 1. 自动判断是否应该展开
# 2. 自动选择展开因子
# 3. 不需要手动写 #pragma
```

### NCU 验证方法

```bash
# 查看指令数
ncu --metrics smsp__inst_executed.sum -o unroll ./program
```

**在 NCU-UI 中查看**：

```
Source 页面：
可以看到展开后的汇编代码

Compute Workload Analysis:
指令数应该增加，但执行时间减少

示例：
未展开：
  循环 100 次 × (1 条 add + 1 条 cmp + 1 条 br) = 300 条指令
  Time: 10 μs

展开 4 次：
  循环 25 次 × (4 条 add + 1 条 cmp + 1 条 br) = 150 条指令
  Time: 7 μs
```

### 注意事项

1. **代码膨胀**
   ```c
   // 展开太多会导致代码量过大
   #pragma unroll 256  // 可能导致指令cache miss
   for (int i = 0; i < 256; i++) {
       // ...
   }

   // 通常展开 4-16 次是最佳的
   ```

2. **寄存器压力**
   ```c
   // 展开增加寄存器使用
   float sum0, sum1, sum2, sum3;  // 4 个寄存器

   // 如果导致寄存器溢出，性能会更差
   // 检查 NCU → Occupancy → Limiting Factor
   ```

3. **何时不展开**
   ```c
   // 大循环不展开
   for (int i = 0; i < 1000000; i++) {  // 太大
       // ...
   }

   // 动态循环不展开
   for (int i = 0; i < N; i++) {  // N 在运行时才知道
       // ...
   }
   ```

4. **编译器自动优化**
   ```bash
   # nvcc -O3 会自动展开一些循环
   nvcc -O3 kernel.cu

   # 可以禁用
   nvcc -O3 -fno-unroll-loops kernel.cu
   ```

---

## 7. Tensor Core

### 原理

**专用硬件**：
- Tensor Core 是专门用于矩阵乘法的硬件单元
- 每个 SM 有 4 个 Tensor Core (A100)
- 一个 Tensor Core 每周期可以执行 256 个 FP16 FMA 操作

**性能对比**（A100）：
```
CUDA Core (FP32): 19.5 TFLOPS
Tensor Core (FP16): 312 TFLOPS  ← 16x ！
Tensor Core (INT8): 624 TOPS   ← 32x ！
```

**支持的数据类型**：
- FP16 (Half Precision)
- BF16 (Brain Float 16)
- TF32 (TensorFloat32) - Ampere+
- INT8, INT4 - Inference

### 何时使用

**NCU 指标判断**：

```
打开 NCU-UI → Details → Compute Workload Analysis

关键指标：
smsp__inst_executed_pipe_tensor.sum

判断：
= 0 且 kernel 是矩阵乘法
→ 未使用 Tensor Core
→ 预期加速：5-20x

> 0 但利用率低
→ 检查 smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
→ 如果 < 50%，可能被其他因素限制
```

**触发条件**：
- 矩阵乘法 (GEMM, Convolution, Attention)
- 数据类型是 FP16/BF16/TF32
- 矩阵维度是 16 的倍数（Ampere 架构）

### CUDA 实现

#### 方法 1：使用 cuBLAS（推荐）

```cpp
#include <cublas_v2.h>

void gemm_cublas_fp16(
    half* A,  // [M, K]
    half* B,  // [K, N]
    half* C,  // [M, N]
    int M, int N, int K
) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 设置使用 Tensor Core
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // FP16 GEMM with Tensor Core
    cublasHgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    cublasDestroy(handle);
}

/*
性能：
cuBLAS FP16 Tensor Core: ~280 TFLOPS (90% 峰值)
手写 FP32: ~15 TFLOPS

加速比：18x
*/
```

#### 方法 2：WMMA API（手动）

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_wmma(
    half* A, half* B, half* C,
    int M, int N, int K
) {
    // WMMA 需要矩阵维度是 16 的倍数
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Tile indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    // Fragments (寄存器中)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // 初始化累加器
    wmma::fill_fragment(c_frag, 0.0f);

    // 主循环
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // 检查边界
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // 加载数据到 fragment
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Tensor Core 计算
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // 存储结果
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// 调用
dim3 gridDim, blockDim;
gridDim.x = (M + 15) / 16;
gridDim.y = (N + 15) / 16;
blockDim.x = 128;

gemm_wmma<<<gridDim, blockDim>>>(A, B, C, M, N, K);

/*
性能：
手写 WMMA: ~200 TFLOPS (64% 峰值)
cuBLAS: ~280 TFLOPS (90% 峰值)

结论：除非特殊需求，否则用 cuBLAS
*/
```

#### 方法 3：TF32（自动，Ampere+）

```cpp
// TF32: FP32 输入，但内部用 Tensor Core 计算
void gemm_cublas_tf32(
    float* A,  // FP32
    float* B,
    float* C,
    int M, int N, int K
) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // TF32 模式（默认开启 on Ampere+）
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    float alpha = 1.0f;
    float beta = 0.0f;

    // 输入是 FP32，但会自动转为 TF32 用 Tensor Core 计算
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    cublasDestroy(handle);
}

/*
性能：
FP32 CUDA Core: 19.5 TFLOPS
TF32 Tensor Core: ~156 TFLOPS (50% of FP16 peak)

精度：
TF32: 10 bits mantissa (vs FP32 23 bits)
对大多数深度学习应用足够
*/
```

### Triton 实现

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Triton 自动使用 Tensor Core！
    # 只要：
    # 1. 数据类型是 FP16/BF16
    # 2. 使用 tl.dot
    # 3. BLOCK 大小合适（16 的倍数）

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])

        # tl.dot 会自动使用 Tensor Core
        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)

# PyTorch 调用
import torch

def matmul_triton(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
    )

    return c

# 使用
a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
c = matmul_triton(a, b)

"""
性能：
Triton (自动 Tensor Core): ~250 TFLOPS
cuBLAS: ~280 TFLOPS

Triton 的优势：
1. 代码简单，自动使用 Tensor Core
2. 性能接近 cuBLAS
3. 易于修改和扩展
"""
```

### NCU 验证方法

```bash
ncu --metrics \
    smsp__inst_executed_pipe_tensor.sum,\
    smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    -o tensor_core ./program
```

**在 NCU-UI 中查看**：

```
Details → Compute Workload Analysis → Instruction Mix

未使用 Tensor Core:
smsp__inst_executed_pipe_tensor: 0  ← 红色警告
smsp__inst_executed_pipe_fp32: 1,000,000

使用 Tensor Core:
smsp__inst_executed_pipe_tensor: 500,000 ✓
smsp__inst_executed_pipe_fp32: 1,000  ← 大幅减少

Tensor Core Utilization Chart:
Peak:   ████████████  100%
Actual: ██████████░░   85%  ← 很好

Time: 100 ms → 5 ms
加速比: 20x ✓
```

### 注意事项

1. **数据类型要求**
   ```cpp
   // ✅ 支持
   half (FP16)
   __nv_bfloat16 (BF16)
   // TF32 (自动，Ampere+)

   // ❌ 不支持
   float (FP32) - 除非用 TF32 mode
   double (FP64)
   ```

2. **矩阵大小要求**
   ```
   Ampere (A100):
   - M, N, K 应该是 16 的倍数
   - 否则需要 padding

   Volta/Turing:
   - M, N, K 应该是 8 的倍数
   ```

3. **精度损失**
   ```
   FP16: 可能溢出/下溢
   → 使用 mixed precision training
   → PyTorch: torch.cuda.amp

   TF32: 降低 mantissa 位数
   → 对大多数深度学习应用影响小
   → 可以禁用: cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)
   ```

4. **何时不用 Tensor Core**
   ```
   - 矩阵很小 (M, N, K < 128)
   - 非常稀疏的矩阵
   - 需要 FP32 精度的科学计算
   - Element-wise 操作（不是矩阵乘法）
   ```

---

## 8. Bank Conflict 避免

### 原理

**Shared Memory 的 Bank 结构**：
```
Shared Memory 被分为 32 个 bank
每个 bank 宽度：4 bytes (32 bits)
每个 bank 每周期：服务 1 个请求

Bank 映射：
address % 32 = bank_id

示例（每个元素 4 bytes）：
data[0]  → Bank 0
data[1]  → Bank 1
...
data[31] → Bank 31
data[32] → Bank 0  (wrap around)
```

**Bank Conflict**：
```
当同一个 warp 的多个 threads 访问同一个 bank 的不同地址时
→ 硬件必须串行化访问
→ 性能下降

示例：
Thread 0: data[0]   → Bank 0
Thread 1: data[32]  → Bank 0  ← Conflict!
Thread 2: data[64]  → Bank 0  ← Conflict!

→ 3 个 thread 访问 Bank 0 的不同地址
→ 需要 3 个周期（串行）
→ 性能降低 3x
```

**广播（Broadcast）**：
```
例外：如果多个 threads 访问同一个地址
→ 硬件会广播数据
→ 不是 conflict

示例：
Thread 0: data[0]
Thread 1: data[0]  ← 同一地址
...
→ 1 个周期完成（广播）✓
```

### 何时使用

**NCU 指标判断**：

```
打开 NCU-UI → Details → Memory Workload Analysis

关键指标：
l1tex__data_bank_conflicts_pipe_lsu.sum

判断：
= 0: 完美 ✓
> 0: 有 bank conflict

Conflict Rate = conflicts / shared_accesses

< 1%: 可接受
1-5%: 应该优化
> 5%: 严重问题

预期收益：
消除 conflict → 1.5-3x (对 Shared Memory 密集的 kernel)
```

**触发条件**：
- 使用 Shared Memory
- 访问模式可能导致 conflict
- 特别是：列访问、stride 访问

### CUDA 实现

#### 案例 1：Matrix Transpose

```c
#define TILE_SIZE 32

// ❌ 有 Bank Conflict
__global__ void transpose_conflict(float* in, float* out, int N) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 读取（行访问，无 conflict）
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    // 写入（列访问，有 conflict！）
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < N && y < N) {
        // tile[threadIdx.x][threadIdx.y]
        // Thread 0: tile[0][0]   → Bank 0
        // Thread 1: tile[0][1]   → Bank 1
        // ...
        // Thread 32: tile[0][32] 不存在（假设 TILE_SIZE=32）

        // 但如果是这样：
        // warp 内 thread 访问：
        // tile[0][tid], tile[1][tid], ..., tile[31][tid]
        // 如果 tid 相同，就是列访问
        // tile[0][0], tile[1][0], ..., tile[31][0]
        // → Bank: 0, 32%32=0, 64%32=0, ...
        // → 所有访问 Bank 0！32-way conflict

        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/*
NCU 结果：
Bank Conflicts: 1,000,000
Time: 5.0 ms
*/

// ✅ 无 Bank Conflict（Padding）
__global__ void transpose_no_conflict(float* in, float* out, int N) {
    // 关键：+1 padding
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 读取
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    // 写入（列访问，但因为 padding 无 conflict）
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < N && y < N) {
        // 现在每行是 33 个元素（不是 32）
        // tile[0][0] → Bank 0
        // tile[1][0] → Bank (33 % 32) = 1
        // tile[2][0] → Bank (66 % 32) = 2
        // ...
        // tile[31][0] → Bank (31*33 % 32) = 31
        // → 所有 thread 访问不同的 bank ✓

        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/*
NCU 结果：
Bank Conflicts: 0 ✓
Time: 1.8 ms

加速比：5.0 / 1.8 = 2.8x
代价：额外 32 × 4 = 128 bytes per block（可接受）
*/
```

#### 案例 2：Reduction

```c
// ❌ 有 Bank Conflict
__global__ void reduce_conflict(float* g_data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? g_data[i] : 0.0f;
    __syncthreads();

    // Reduction with conflicts
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            // sdata[0] += sdata[1]
            // sdata[2] += sdata[3]
            // ...
            // Thread 0 访问 sdata[0], sdata[1] → Bank 0, 1
            // Thread 2 访问 sdata[2], sdata[3] → Bank 2, 3
            // 看起来没问题？

            // 但随着 s 增大：
            // s=16 时：
            // Thread 0: sdata[0] += sdata[16]
            // Thread 32: sdata[32] += sdata[48]
            // sdata[32] → Bank 0
            // → 有 conflict

            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// ✅ 无 Bank Conflict
__global__ void reduce_no_conflict(float* g_data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? g_data[i] : 0.0f;
    __syncthreads();

    // 改进的 reduction 模式
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 前半部分 thread 活跃
            // Thread 0: sdata[0] += sdata[128]
            // Thread 1: sdata[1] += sdata[129]
            // ...
            // Thread 127: sdata[127] += sdata[255]

            // Bank 访问：
            // sdata[0], sdata[128] → Bank 0, 0 ← 同一 bank 不同地址？
            // 但 128 % 32 = 0，所以是同一 bank

            // 实际上还是有 conflict...

            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// ✅✅ 真正无 Conflict（使用 sequential addressing）
__global__ void reduce_sequential(float* g_data, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 先在 global memory 做一次 reduction
    sdata[tid] = g_data[i] + g_data[i + blockDim.x];
    __syncthreads();

    // Sequential addressing（关键！）
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/*
性能对比：
Conflict:    2.5 ms, Bank Conflicts: 50,000
No Conflict: 1.0 ms, Bank Conflicts: 0
加速比：2.5x
*/
```

### Triton 实现

```python
# Triton 自动避免 bank conflict！
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Triton 编译器会自动：
    # 1. 分析访问模式
    # 2. 插入 padding 如果需要
    # 3. 重组数据布局避免 conflict

    # 程序员不需要担心 bank conflict
    # ...
```

### NCU 验证方法

```bash
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_shared.sum \
    -o bank_conflict ./program
```

**在 NCU-UI 中查看**：

```
Details → Memory Workload Analysis

Bank Conflicts (优化前):
l1tex__data_bank_conflicts_pipe_lsu: 1,000,000
Shared Memory Accesses: 10,000,000
Conflict Rate: 10%  ← 严重

Bank Conflicts (优化后):
l1tex__data_bank_conflicts_pipe_lsu: 0 ✓
Conflict Rate: 0%
```

### 注意事项

1. **Padding 的代价**
   ```c
   __shared__ float tile[32][33];  // +1 column

   // 代价：
   // 每个 block 额外使用 32 × 4 = 128 bytes
   // A100: Shared Memory 总共 96 KB per block
   // 128 bytes 占 0.13%，完全可接受
   ```

2. **Warp Shuffle 替代**
   ```c
   // 对于 reduction，可以用 warp shuffle 避免 shared memory
   __device__ float warp_reduce_sum(float val) {
       for (int offset = 16; offset > 0; offset /= 2) {
           val += __shfl_down_sync(0xffffffff, val, offset);
       }
       return val;
   }

   // 无 shared memory → 无 bank conflict
   // 且更快
   ```

3. **结构体数组**
   ```c
   // ❌ 可能有 conflict
   struct Point {
       float x, y, z;
   };
   __shared__ Point points[32];

   // points[0].x → offset 0 → Bank 0
   // points[1].x → offset 12 → Bank 12/4=3
   // ...
   // 可能有 conflict

   // ✅ 分离数组
   __shared__ float x[32], y[32], z[32];
   // 更好的访问模式
   ```

4. **检测工具**
   ```bash
   # cuda-memcheck 可以检测 bank conflict
   cuda-memcheck --tool memcheck ./program

   # 但 NCU 更精确
   ```

---

## 9. Warp Divergence 优化

### 原理

**SIMT 执行模型**：
```
GPU 以 warp (32 threads) 为单位执行
同一时刻，warp 内所有 threads 执行相同指令 (SIMD)
但可以通过 mask 关闭部分 thread

分支：
if (condition) {
    // Branch A
} else {
    // Branch B
}

执行过程：
1. 评估 condition for all 32 threads
2. 执行 Branch A，mask 掉 condition=false 的 threads
3. 执行 Branch B，mask 掉 condition=true 的 threads
4. 总耗时 = Time(A) + Time(B)
```

**Divergence 的代价**：
```c
// 示例
if (tid % 2 == 0) {
    // Branch A: 10 instructions
} else {
    // Branch B: 10 instructions
}

// 无 divergence: 10 instructions
// 有 divergence: 10 + 10 = 20 instructions
// 效率：50%
```

### 何时使用

**NCU 指标判断**：

```
打开 NCU-UI → Details → Compute Workload Analysis

关键指标：
smsp__thread_inst_executed_per_inst_executed

理想值：32 (warp 所有 threads 都执行)
实际值：< 32 → 有 divergence

效率 = actual / 32 × 100%

> 90%: 很好
80-90%: 可接受
< 80%: 应该优化

预期收益：
消除 divergence → 1.5-3x (对 divergence 严重的 kernel)
```

**触发条件**：
- 有 if/else 分支
- Threads 走不同分支
- 特别是：thread-dependent 条件

### CUDA 实现

#### 案例 1：条件计算

```c
// ❌ 严重 Divergence
__global__ void compute_divergence(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        // 每个 thread 根据 tid 走不同分支
        if (tid % 2 == 0) {
            // 一半 threads
            data[tid] = expensive_compute_even(data[tid]);
        } else {
            // 另一半 threads
            data[tid] = expensive_compute_odd(data[tid]);
        }
    }
}

/*
NCU 结果：
Thread Inst Per Inst: 16
Efficiency: 50%
*/

// ✅ 减少 Divergence（数据重组）
// 预处理：将 even 和 odd 分开

__global__ void compute_no_divergence_even(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N / 2) {
        data[tid * 2] = expensive_compute_even(data[tid * 2]);
    }
}

__global__ void compute_no_divergence_odd(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N / 2) {
        data[tid * 2 + 1] = expensive_compute_odd(data[tid * 2 + 1]);
    }
}

// 调用
compute_no_divergence_even<<<grid, block>>>(data, N);
compute_no_divergence_odd<<<grid, block>>>(data, N);

/*
NCU 结果：
Thread Inst Per Inst: 32
Efficiency: 100% ✓
*/
```

#### 案例 2：Warp-level 条件

```c
// ❌ Thread-level 分支（divergence）
__global__ void bad_branch(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        if (tid % 32 < 16) {  // 每个 warp 内一半 threads
            data[tid] = compute_a(data[tid]);
        } else {
            data[tid] = compute_b(data[tid]);
        }
    }
}

// ✅ Warp-level 分支（无 divergence）
__global__ void good_branch(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        int warp_id = tid / 32;  // Warp ID

        if (warp_id % 2 == 0) {  // 整个 warp 走同一分支
            data[tid] = compute_a(data[tid]);
        } else {
            data[tid] = compute_b(data[tid]);
        }
    }
}

/*
原理：
bad_branch: warp 内部分支 → divergence
good_branch: 不同 warp 走不同分支 → 无 divergence
*/
```

#### 案例 3：Predication（谓词）

```c
// ❌ 分支
__global__ void with_branch(float* data, int N, float threshold) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        if (data[tid] > threshold) {
            data[tid] = data[tid] * 2.0f;
        } else {
            data[tid] = data[tid] * 0.5f;
        }
    }
}

// ✅ 无分支（Predication）
__global__ void without_branch(float* data, int N, float threshold) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        float val = data[tid];

        // 两个计算都执行
        float result_true = val * 2.0f;
        float result_false = val * 0.5f;

        // 用三元运算符选择（编译器可能优化成 predication）
        data[tid] = (val > threshold) ? result_true : result_false;

        // 或者明确使用 select
        // data[tid] = fmaxf(
        //     (val > threshold) ? result_true : -FLT_MAX,
        //     (val > threshold) ? -FLT_MAX : result_false
        // );
    }
}

/*
注意：
- 只有当两个分支都很简单时才值得
- 如果分支很复杂，两个都执行反而更慢
- 需要 benchmark 验证
*/
```

### Triton 实现

```python
# Triton 对 divergence 的处理更智能
@triton.jit
def kernel_with_branch(
    data_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data = tl.load(data_ptr + offsets, mask=mask)

    # Triton 会尝试优化分支
    # 如果可能，会用 predication
    result = tl.where(data > 0, data * 2.0, data * 0.5)

    # tl.where 通常编译成 predicated instruction
    # 没有分支

    tl.store(data_ptr + offsets, result, mask=mask)
```

### NCU 验证方法

```bash
ncu --metrics \
    smsp__thread_inst_executed_per_inst_executed,\
    smsp__sass_branch_targets.sum \
    -o divergence ./program
```

**在 NCU-UI 中查看**：

```
Details → Compute Workload Analysis

优化前：
Thread Inst Per Inst: 16
Efficiency: 50%
Branch Instructions: 100,000

优化后：
Thread Inst Per Inst: 31.5  ← 接近 32
Efficiency: 98% ✓
Branch Instructions: 10,000  ← 减少
```

### 注意事项

1. **不是所有分支都会 divergence**
   ```c
   // ✓ 无 divergence：所有 threads 走相同分支
   if (blockIdx.x == 0) {
       // ...
   }

   // ✗ 有 divergence：warp 内不同 threads 走不同分支
   if (threadIdx.x % 2 == 0) {
       // ...
   }
   ```

2. **Early Exit 也可能导致 divergence**
   ```c
   // ❌ Divergence
   if (tid >= N) return;  // 部分 threads 提前退出

   // ✅ 更好：用 mask
   if (tid < N) {
       // 所有 threads 都执行到这里
       // 只是 tid >= N 的 threads 不做实际操作
   }
   ```

3. **Loop Divergence**
   ```c
   // ❌ 不同 threads 循环次数不同
   for (int i = 0; i < data[tid]; i++) {  // data[tid] 不同
       // ...
   }

   // ✅ 统一循环次数
   int max_iter = ...; // 提前计算最大值
   for (int i = 0; i < max_iter; i++) {
       if (i < data[tid]) {  // 用条件控制
           // ...
       }
   }
   ```

4. **Trade-off**
   ```
   有时消除 divergence 的代价 > 收益

   示例：
   - Branch A: 100 instructions
   - Branch B: 10 instructions
   - 50% threads 走 A, 50% 走 B

   有 divergence: 100 + 10 = 110 instructions
   无 divergence 但都执行: 200 instructions (100 × 2)

   → 保持 divergence 更好
   ```

---

## 10. Occupancy 优化

### 原理

**Occupancy 定义**：
```
Occupancy = Active Warps / Maximum Warps

A100 每个 SM：
- Maximum Warps: 64
- Maximum Threads: 2048
- Maximum Blocks: 32

Occupancy 例子：
- 32 active warps / 64 max = 50% Occupancy
```

**为什么 Occupancy 重要**：
```
高 Occupancy → 更多 warps 可以切换
→ 可以隐藏内存延迟

示例：
Occupancy 25% (16 warps):
- Warp 0 等待内存 (400 cycles)
- 只有 15 个其他 warps 可以执行
- 可能仍有 idle 时间

Occupancy 100% (64 warps):
- Warp 0 等待内存
- 有 63 个其他 warps 可以执行
- 更容易隐藏延迟
```

**但高 Occupancy 不是万能的**：
```
如果不是等待延迟，高 Occupancy 无用

示例：
- Compute-bound: 等待计算完成，不是延迟
  → 更多 warps 也在等待
  → Occupancy 提升无效

- Memory bandwidth-bound: 带宽饱和
  → 更多 warps 只会增加带宽竞争
  → Occupancy 提升可能有害
```

### 何时使用

**NCU 指标判断**：

```
打开 NCU-UI → Details → Occupancy

关键指标：
1. Theoretical Occupancy: 理论最大值
2. Achieved Occupancy: 实际值

同时检查 Warp State Statistics:
stall_not_selected: warp 因 occupancy 不足而停顿

判断：
Theoretical = 100%, Achieved = 50%
且 stall_not_selected > 30%
→ 应该提高 Occupancy

Theoretical = 50%, Achieved = 50%
→ 检查 Limiting Factor，解决限制因素

stall_not_selected < 20%
→ Occupancy 已经足够，不是瓶颈
```

**触发条件**：
- Memory-bound kernel
- 有显著的 stall_long_scoreboard
- stall_not_selected > 30%

### CUDA 实现

#### Occupancy Limiter 1：寄存器

```c
// ❌ 使用太多寄存器
__global__ void too_many_registers(...) {
    float a, b, c, d, e, f, g, h;  // 8 个寄存器
    float i, j, k, l, m, n, o, p;  // 16 个寄存器
    float q, r, s, t, u, v, w, x;  // 24 个寄存器
    // ...
    // 总共 > 64 个寄存器

    // 复杂计算
    // ...
}

/*
NCU 结果：
Occupancy: 25%
Limiting Factor: Registers
每个 thread 用了 80 个寄存器
→ 每个 block 只能有 32 warps (256 threads)
→ Occupancy 降低
*/

// ✅ 减少寄存器使用
__global__ void __launch_bounds__(256, 4)  // 提示编译器
fewer_registers(...) {
    // 重用寄存器
    float temp;

    temp = compute_a(...);
    output_a = temp;

    temp = compute_b(...);  // 重用 temp
    output_b = temp;

    // ...
}

// 或者编译选项
// nvcc -maxrregcount=32 kernel.cu

/*
NCU 结果：
Occupancy: 100%
Registers per thread: 32 ← 减少
*/
```

#### Occupancy Limiter 2：Shared Memory

```c
// ❌ 使用太多 Shared Memory
__global__ void too_much_shared(...) {
    __shared__ float data[64 * 1024 / sizeof(float)];  // 64 KB

    // A100: 每个 block 最多 96 KB Shared Memory
    // 但如果用了 64 KB
    // → 每个 SM 只能同时运行 1-2 个 blocks
    // → Occupancy 降低
}

/*
NCU 结果：
Occupancy: 12.5%
Limiting Factor: Shared Memory
*/

// ✅ 减少 Shared Memory 或接受较低 Occupancy
__global__ void balanced_shared(...) {
    __shared__ float data[32][32];  // 4 KB

    // 或者，如果必须用 64 KB:
    // 接受较低的 Occupancy，因为 Shared Memory 带来的收益
    // > Occupancy 损失
}
```

#### Occupancy Limiter 3：Block Size

```c
// ❌ Block size 太小
dim3 block(32, 1);  // 只有 32 threads per block
dim3 grid((N + 31) / 32);

kernel<<<grid, block>>>(...);

/*
问题：
- 每个 block 只有 1 warp
- 即使有 64 个 blocks，SM 最多同时运行 32 blocks
- → 只有 32 warps active
- → Occupancy = 32/64 = 50%
*/

// ✅ 增大 Block Size
dim3 block(256, 1);  // 256 threads = 8 warps
dim3 grid((N + 255) / 256);

kernel<<<grid, block>>>(...);

/*
现在：
- 每个 block 有 8 warps
- 8 blocks × 8 warps = 64 warps
- Occupancy = 100%
*/
```

#### 使用 Occupancy Calculator

```c
// CUDA 提供的 API
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxActiveBlocks,
    kernel,
    blockSize,
    dynamicSharedMemSize
);

printf("Max active blocks: %d\n", maxActiveBlocks);

// 或者运行时查询
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,
    &blockSize,
    kernel,
    dynamicSharedMemSize,
    maxBlockSize
);

printf("Recommended block size: %d\n", blockSize);
```

### Triton 实现

```python
# Triton 自动选择 block size
# 可以通过 num_warps 参数控制

@triton.jit
def kernel(
    data_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # ...
    pass

# 调用时指定
kernel[grid](
    data,
    BLOCK_SIZE=1024,
    num_warps=4,  # 每个 block 4 warps = 128 threads
                  # Triton 会自动选择最佳配置
)

# Auto-tuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel_autotuned(data_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ...
    pass

# Triton 会自动选择最快的配置
```

### NCU 验证方法

```bash
ncu --section Occupancy -o occupancy ./program
```

**在 NCU-UI 中查看**：

```
Details → Occupancy

Theoretical Occupancy: 100%
Achieved Occupancy: 62%

Limiting Factor: Shared Memory
- Shared Memory per block: 48 KB
- Limit: 96 KB
- Blocks per SM (limited by SM): 2
- Theoretical: 100% (resources足够)
- Achieved: 62% (其他因素)

同时检查 Warp State Statistics:
stall_not_selected: 15%  ← 还行

结论：
虽然 Occupancy 62%，但 stall_not_selected 只有 15%
→ Occupancy 已经足够
→ 不需要进一步优化
```

### 注意事项

1. **Occupancy 不是目标**
   ```
   目标是性能，不是 Occupancy

   有时低 Occupancy 更快：
   - 使用 Shared Memory 提高数据复用
   - 虽然 Occupancy 降低，但整体更快

   示例：
   Kernel A: Occupancy 100%, Time 10 ms
   Kernel B: Occupancy 50%, Time 6 ms ← 更好！
   ```

2. **检查是否真的需要**
   ```
   NCU → Warp State Statistics → stall_not_selected

   < 20%: Occupancy 已经足够
   > 40%: 需要提高 Occupancy
   ```

3. **Trade-off**
   ```c
   // Shared Memory vs Occupancy

   // 选项 A: 小 Shared Memory，高 Occupancy
   __shared__ float data[16][16];  // 1 KB
   // Occupancy: 100%
   // 但数据复用少

   // 选项 B: 大 Shared Memory，低 Occupancy
   __shared__ float data[64][64];  // 16 KB
   // Occupancy: 50%
   // 但数据复用多

   // 需要 benchmark 决定
   ```

4. **动态调整**
   ```c
   // 根据硬件动态选择配置
   int device;
   cudaGetDevice(&device);

   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, device);

   int sharedMemPerBlock = prop.sharedMemPerBlock;
   int maxThreadsPerBlock = prop.maxThreadsPerBlock;

   // 根据硬件选择 block size
   int blockSize = (sharedMemPerBlock > 48*1024) ? 512 : 256;
   ```

---

## 总结：优化技术选择决策树

```
                    开始优化
                        │
                        ▼
                  运行 NCU Profile
                        │
            ┌───────────┴───────────┐
            │                       │
      Memory-bound           Compute-bound
            │                       │
            ├─ Sectors > 1.5?      ├─ Tensor Core = 0?
            │  → Coalescing        │  → 使用 Tensor Core (5-20x)
            │                      │
            ├─ L2 Hit < 50%?       ├─ Divergence 严重?
            │  → Fusion (3-10x)    │  → 减少 Divergence (1.5-3x)
            │  → Shared Memory     │
            │                      ├─ Issue Eff < 60%?
            ├─ Coalesced but       │  → ILP (1.3-2x)
            │  BW < 90%?           │  → Loop Unrolling
            │  → Vectorization     │
            │                      └─ Optimized!
            ├─ Bank Conflicts?
            │  → Padding
            │
            ├─ stall_not_selected>40%?
            │  → Occupancy
            │
            └─ Optimized!
```

### 优化优先级

| 优先级 | 技术 | 适用场景 | 预期加速 | 难度 |
|-------|-----|---------|---------|------|
| ⭐⭐⭐⭐⭐ | Memory Coalescing | Sectors > 1.5 | 5-30x | 容易 |
| ⭐⭐⭐⭐⭐ | Tensor Core | GEMM + FP16 | 5-20x | 容易 |
| ⭐⭐⭐⭐⭐ | Kernel Fusion | 多个小 kernel | 3-10x | 中等 |
| ⭐⭐⭐⭐ | Shared Memory | 数据复用 | 2-5x | 中等 |
| ⭐⭐⭐⭐ | Vectorization | Memory-bound | 1.5-2x | 容易 |
| ⭐⭐⭐ | Bank Conflict | 用 Shared Memory | 1.5-3x | 容易 |
| ⭐⭐⭐ | Divergence | 有分支 | 1.5-3x | 中等 |
| ⭐⭐⭐ | ILP | Compute-bound | 1.3-2x | 中等 |
| ⭐⭐ | Loop Unrolling | 小循环 | 1.2-1.5x | 容易 |
| ⭐⭐ | Occupancy | stall_not_selected高 | 1.2-2x | 中等 |

### CUDA vs Triton

| 方面 | CUDA | Triton |
|-----|------|--------|
| **Coalescing** | 手动保证 | 自动优化 ✓ |
| **Vectorization** | 手动 float4 | 自动 ✓ |
| **Shared Memory** | 手动管理 | 自动管理 ✓ |
| **Fusion** | 手动实现 | 容易实现 |
| **ILP** | 手动展开 | 自动优化 ✓ |
| **Tensor Core** | WMMA API | tl.dot 自动 ✓ |
| **Bank Conflict** | 手动 padding | 自动避免 ✓ |
| **Divergence** | 手动优化 | 自动 predication ✓ |
| **Occupancy** | 手动调优 | 自动 tuning ✓ |
| **性能上限** | 最高 (100%) | 很高 (90-95%) |
| **开发效率** | 低 | 高 ✓ |

**建议**：
- **原型和实验**：用 Triton（快速迭代）
- **生产环境关键 kernel**：用 CUDA（极致性能）或 Triton（足够好）
- **标准操作 (GEMM, Conv)**：用 cuBLAS/cuDNN（无需重新发明轮子）

---

**完结！**这个手册涵盖了 GPU 优化的 10 大核心技术，每个技术都有详细的原理、使用场景、CUDA/Triton 实现、NCU 验证方法。