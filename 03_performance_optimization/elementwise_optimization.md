# Element-wise 算子优化完全指南

## 目录
1. [Element-wise 算子理解](#element-wise-算子理解)
2. [性能特点分析](#性能特点分析)
3. [优化策略](#优化策略)
4. [实战案例](#实战案例)
5. [高级技巧](#高级技巧)

---

## Element-wise 算子理解

### 什么是 Element-wise 算子？

**定义**：对输入张量的每个元素独立执行相同操作，输出张量的每个元素只依赖于输入张量对应位置的元素。

```python
# Element-wise 操作示例
C = A + B           # 逐元素加法
C = A * B           # 逐元素乘法
C = relu(A)         # ReLU: max(0, A)
C = sigmoid(A)      # Sigmoid 激活
C = A + alpha       # 标量加法
C = A * B + C       # 融合操作
```

**数学表达**：
```
C[i] = f(A[i], B[i], ...)
```
其中 `i` 是元素索引，`f` 是操作函数。

### Element-wise vs 其他算子类型

| 算子类型 | 数据依赖 | Arithmetic Intensity | 瓶颈 |
|---------|---------|---------------------|------|
| **Element-wise** | 单个元素 | 很低 (~0.25) | Memory-bound |
| **Reduction** | 多个元素 | 低 (~1) | Memory-bound |
| **MatMul** | 整行/列 | 很高 (50-100+) | Compute-bound |
| **Convolution** | 局部区域 | 高 (10-50) | Compute-bound |

**关键特点**：
```
Arithmetic Intensity = FLOPs / Bytes Accessed

Element-wise: AI = O(1) / O(4-8 bytes) ≈ 0.25 FLOPs/Byte
```

这意味着 Element-wise 算子是典型的 **Memory-Bound**（内存受限）！

---

## 性能特点分析

### 1. Roofline 分析

```
Performance (TFLOPS)
    |
    |         Compute Roof (100 TFLOPS for A100)
    |
    |   MatMul *
    |         /
    |        /
    |       /
    |      / Memory Roof
    |     /
    |    * Element-wise (在这里！)
    |   /
    |  /
    |_/___________________________
         Arithmetic Intensity

Element-wise 在左下角 → 完全受内存带宽限制
```

### 2. 理论性能分析

**对于 NVIDIA A100：**
- Peak Memory Bandwidth: 1555 GB/s
- Peak Compute: 19.5 TFLOPS (FP32)

**向量加法 (C = A + B)**：
```
操作：每个元素 1 次加法 = 1 FLOP
访问：读 A (4 bytes) + 读 B (4 bytes) + 写 C (4 bytes) = 12 bytes

Arithmetic Intensity = 1 FLOP / 12 bytes ≈ 0.083 FLOPs/Byte

理论峰值性能 = Min(
    1555 GB/s * 0.083 = 129 GFLOPS,  ← Memory 限制
    19500 GFLOPS                      ← Compute 限制
)
= 129 GFLOPS (只用了 0.66% 的计算能力！)
```

**结论**：Element-wise 算子的性能完全由内存带宽决定！

### 3. 实际性能瓶颈

```bash
# 使用 NCU 分析
ncu --section SpeedOfLight ./element_wise_kernel
```

**典型输出**：
```
Speed of Light
  SM Throughput:       5%     ← 计算单元几乎空闲
  Memory Throughput:   85%    ← 内存带宽接近饱和

结论：Memory-bound！
```

---

## 优化策略

### 策略 1: 向量化访问

**原理**：一次读取多个元素，提高内存事务效率。

#### 不好的实现
```cpp
__global__ void add_naive(float* C, const float* A, const float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // 每次读/写 4 bytes
    }
}
```

**问题**：每个线程每次只处理 4 bytes，内存事务效率低。

#### 优化版本 1：float4 向量化
```cpp
__global__ void add_vectorized(float* C, const float* A, const float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;  // 每个线程处理 4 个元素

    if (vec_idx + 3 < N) {
        // 一次读取 16 bytes (4个float)
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];

        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        // 一次写入 16 bytes
        reinterpret_cast<float4*>(C)[idx] = c;
    }
}
```

**性能提升**：1.5-2x

#### 优化版本 2：更高的向量化
```cpp
__global__ void add_vectorized_8(float* C, const float* A, const float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 8;

    if (base + 7 < N) {
        // 使用两个 float4，一次处理 8 个元素
        float4 a0 = reinterpret_cast<const float4*>(A)[idx * 2];
        float4 a1 = reinterpret_cast<const float4*>(A)[idx * 2 + 1];
        float4 b0 = reinterpret_cast<const float4*>(B)[idx * 2];
        float4 b1 = reinterpret_cast<const float4*>(B)[idx * 2 + 1];

        float4 c0, c1;
        c0.x = a0.x + b0.x; c0.y = a0.y + b0.y;
        c0.z = a0.z + b0.z; c0.w = a0.w + b0.w;
        c1.x = a1.x + b1.x; c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z; c1.w = a1.w + b1.w;

        reinterpret_cast<float4*>(C)[idx * 2] = c0;
        reinterpret_cast<float4*>(C)[idx * 2 + 1] = c1;
    }
}
```

**性能提升**：2-2.5x（相对 naive）

### 策略 2: 算子融合

**原理**：将多个 element-wise 操作融合到一个 kernel，减少内存访问次数。

#### 融合前：3 个 kernel
```cpp
// Kernel 1: ReLU
__global__ void relu(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Kernel 2: Add bias
__global__ void add_bias(float* data, float bias, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += bias;
    }
}

// Kernel 3: Scale
__global__ void scale(float* data, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= scale;
    }
}

// 使用：3 次 kernel 启动，3 次内存读写
relu<<<grid, block>>>(data, N);        // Read + Write
add_bias<<<grid, block>>>(data, bias, N);  // Read + Write
scale<<<grid, block>>>(data, scale, N);    // Read + Write
```

**开销分析**：
```
内存访问：3 * (Read 4B + Write 4B) = 24 bytes/element
Kernel 启动：3 * ~10 μs = ~30 μs
```

#### 融合后：1 个 kernel
```cpp
__global__ void fused_relu_bias_scale(
    float* data, float bias, float scale, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];     // 只读一次
        val = fmaxf(0.0f, val);    // ReLU
        val += bias;                // Add
        val *= scale;               // Scale
        data[idx] = val;            // 只写一次
    }
}
```

**开销分析**：
```
内存访问：1 * (Read 4B + Write 4B) = 8 bytes/element
Kernel 启动：1 * ~10 μs = ~10 μs

带宽节省：(24 - 8) / 24 = 66.7%
性能提升：~3x
```

#### 融合 + 向量化
```cpp
__global__ void fused_relu_bias_scale_vectorized(
    float* data, float bias, float scale, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < N) {
        float4 val = reinterpret_cast<const float4*>(data)[idx];

        // 融合操作
        val.x = fmaxf(0.0f, val.x) + bias;
        val.y = fmaxf(0.0f, val.y) + bias;
        val.z = fmaxf(0.0f, val.z) + bias;
        val.w = fmaxf(0.0f, val.w) + bias;

        val.x *= scale;
        val.y *= scale;
        val.z *= scale;
        val.w *= scale;

        reinterpret_cast<float4*>(data)[idx] = val;
    }
}
```

**性能提升**：5-8x（相对原始 3 个 kernel）

### 策略 3: Grid-Stride Loop

**原理**：让每个线程处理多个元素，提高数据复用和寄存器利用率。

```cpp
__global__ void add_grid_stride(float* C, const float* A, const float* B, int N) {
    // 计算 grid 的总线程数
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 每个线程处理多个元素
    for (int i = tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

**优势**：
- 适应任意大小的输入（不需要精确计算 grid size）
- 提高指令级并行（ILP）
- 更好的寄存器复用

**配合向量化**：
```cpp
__global__ void add_grid_stride_vectorized(
    float* C, const float* A, const float* B, int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 向量化 + Grid-stride
    for (int i = tid * 4; i + 3 < N; i += stride * 4) {
        int idx = i / 4;
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];

        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        reinterpret_cast<float4*>(C)[idx] = c;
    }
}
```

### 策略 4: 使用快速数学函数

```cpp
__global__ void activation_optimized(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = data[idx];

        // ❌ 慢
        // x = expf(x);

        // ✅ 快 (使用内置快速函数)
        x = __expf(x);

        // 其他快速函数
        // __logf(x)    - 快速对数
        // __sinf(x)    - 快速正弦
        // __cosf(x)    - 快速余弦
        // __powf(x, y) - 快速幂
        // rsqrtf(x)    - 快速平方根倒数

        data[idx] = x;
    }
}
```

**性能提升**：1.5-3x（对于超越函数）

**编译选项**：
```bash
# 使用快速数学
nvcc -use_fast_math kernel.cu

# 或者单独指定
nvcc --ftz=true --prec-div=false --prec-sqrt=false kernel.cu
```

### 策略 5: 内存访问模式优化

#### 确保对齐
```cpp
__global__ void add_aligned(float* C, const float* A, const float* B, int N) {
    // 检查指针对齐
    assert(((uintptr_t)A % 16) == 0);  // 16-byte 对齐
    assert(((uintptr_t)B % 16) == 0);
    assert(((uintptr_t)C % 16) == 0);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < N) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        // ...
    }
}
```

**分配对齐内存**：
```cpp
// 使用 cudaMalloc (自动 256-byte 对齐)
cudaMalloc(&d_data, N * sizeof(float));

// 或者显式对齐
size_t pitch;
cudaMallocPitch(&d_data, &pitch, width * sizeof(float), height);
```

---

## 实战案例

### 案例 1: GELU 激活函数优化

GELU（Gaussian Error Linear Unit）广泛用于 Transformer：

```
GELU(x) = x * Φ(x)
       = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

#### Naive 实现
```cpp
__global__ void gelu_naive(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];
        float x3 = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        float tanh_val = tanhf(inner);
        out[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}
```

**NCU 分析**：
```
SM Throughput:     15%
Memory Throughput: 85%
Instruction Throughput: 45% (tanh 很慢)
```

#### 优化版本 1：快速近似
```cpp
// Tanh 快速近似：tanh(x) ≈ x / (1 + |x|) for small x
__device__ float fast_tanh(float x) {
    float x2 = x * x;
    float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
}

__global__ void gelu_fast(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];
        const float c = sqrtf(2.0f / M_PI);
        float x3 = x * x * x;
        float inner = c * (x + 0.044715f * x3);
        float tanh_val = fast_tanh(inner);
        out[idx] = 0.5f * x * (1.0f + tanh_val);
    }
}
```

#### 优化版本 2：向量化 + FMA
```cpp
__global__ void gelu_optimized(float* out, const float* in, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < N) {
        const float c = sqrtf(2.0f / M_PI);
        const float alpha = 0.044715f;

        float4 x = reinterpret_cast<const float4*>(in)[idx / 4];
        float4 result;

        // 使用 FMA (Fused Multiply-Add)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = (&x.x)[i];
            float val2 = val * val;
            float val3 = val2 * val;

            // 使用 __fmaf_rn 进行 FMA
            float inner = __fmaf_rn(c * alpha, val3, c * val);
            float tanh_val = fast_tanh(inner);
            (&result.x)[i] = 0.5f * val * __fmaf_rn(1.0f, tanh_val, 1.0f);
        }

        reinterpret_cast<float4*>(out)[idx / 4] = result;
    }
}
```

**性能对比**：
```
Naive:      12.5 ms
Fast approx: 8.2 ms (1.5x)
Vectorized:  3.1 ms (4x)
```

### 案例 2: LayerNorm 的 Element-wise 部分

LayerNorm 包含 element-wise 和 reduction 操作：

```python
# LayerNorm 步骤
mean = x.mean(dim=-1, keepdim=True)          # Reduction
var = ((x - mean) ** 2).mean(dim=-1)         # Reduction
x_norm = (x - mean) / sqrt(var + eps)        # Element-wise
out = gamma * x_norm + beta                   # Element-wise
```

#### 融合 element-wise 部分
```cpp
__global__ void layernorm_elementwise_fused(
    float* out,
    const float* in,
    const float* gamma,
    const float* beta,
    const float* mean,  // 已经计算好的 mean
    const float* var,   // 已经计算好的 var
    int N, int D
) {
    int row = blockIdx.x;
    int col = (blockIdx.y * blockDim.x + threadIdx.x) * 4;

    if (col + 3 < D) {
        int idx = row * D + col;
        int vec_idx = idx / 4;

        float m = mean[row];
        float v = var[row];
        float inv_std = rsqrtf(v + 1e-5f);  // 使用快速平方根倒数

        float4 x = reinterpret_cast<const float4*>(in)[vec_idx];
        float4 g = reinterpret_cast<const float4*>(gamma)[col / 4];
        float4 b = reinterpret_cast<const float4*>(beta)[col / 4];

        float4 result;
        result.x = __fmaf_rn(g.x, (x.x - m) * inv_std, b.x);
        result.y = __fmaf_rn(g.y, (x.y - m) * inv_std, b.y);
        result.z = __fmaf_rn(g.z, (x.z - m) * inv_std, b.z);
        result.w = __fmaf_rn(g.w, (x.w - m) * inv_std, b.w);

        reinterpret_cast<float4*>(out)[vec_idx] = result;
    }
}
```

**优化点**：
- 使用 `rsqrtf` 代替 `1.0f / sqrtf`
- 向量化访问
- 使用 FMA 指令
- 融合归一化和仿射变换

---

## 高级技巧

### 技巧 1: 使用 __restrict__ 关键字

帮助编译器优化（告诉编译器指针不会 alias）：

```cpp
__global__ void add_restrict(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int N
) {
    // 编译器可以更激进地优化
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### 技巧 2: 模板元编程

```cpp
template<typename T, int VECTOR_SIZE>
__global__ void elementwise_template(T* out, const T* in, int N) {
    using VecType = typename std::conditional<
        VECTOR_SIZE == 4, float4,
        typename std::conditional<VECTOR_SIZE == 2, float2, float>::type
    >::type;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * VECTOR_SIZE;

    if (vec_idx + VECTOR_SIZE - 1 < N) {
        VecType val = reinterpret_cast<const VecType*>(in)[idx];
        // 操作...
        reinterpret_cast<VecType*>(out)[idx] = val;
    }
}
```

### 技巧 3: Warp-level 优化

对于非常小的 element-wise 操作，可以使用 warp shuffle：

```cpp
__global__ void broadcast_add_warp_optimized(
    float* out, const float* in, float scalar, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp 内广播 scalar（如果 scalar 是数组的一个元素）
    // float val = __shfl_sync(0xffffffff, scalar, 0);

    if (idx < N) {
        out[idx] = in[idx] + scalar;
    }
}
```

### 技巧 4: 使用 CUDA Graphs

对于重复执行的 element-wise 操作序列：

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

// 捕获操作序列
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

relu<<<grid, block, 0, stream>>>(data, N);
add_bias<<<grid, block, 0, stream>>>(data, bias, N);
scale<<<grid, block, 0, stream>>>(data, scale, N);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// 重复执行（开销极低）
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);
}
```

**性能提升**：2-3x（对于小 kernel 序列）

---

## Triton 中的 Element-wise 优化

Triton 自动处理很多优化，但仍有技巧：

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_gelu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # GELU 近似
    # Triton 会自动向量化和优化
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)

    # Triton 的 tanh 会被优化
    tanh_val = tl.libdevice.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)

    tl.store(out_ptr + offsets, out, mask=mask)
```

**Triton 的优势**：
- 自动选择最优 BLOCK_SIZE
- 自动向量化
- 自动内存 coalescing
- JIT 优化

---

## 性能检查清单

### Element-wise 优化清单 ✓

- [ ] 使用向量化访问 (float4, float2)
- [ ] 融合多个操作到单个 kernel
- [ ] 使用快速数学函数 (`__expf`, `rsqrtf`)
- [ ] 确保内存对齐
- [ ] 使用 `__restrict__` 关键字
- [ ] 考虑 Grid-stride loop
- [ ] 使用 FMA 指令
- [ ] 对于序列操作，考虑 CUDA Graphs

### 性能验证 ✓

```cpp
// 计算带宽利用率
float theoretical_bandwidth = 1555e9;  // A100: 1555 GB/s
float bytes_accessed = N * (num_reads + num_writes) * sizeof(float);
float time_seconds = time_ms / 1000.0f;
float actual_bandwidth = bytes_accessed / time_seconds;
float utilization = actual_bandwidth / theoretical_bandwidth * 100;

printf("Bandwidth Utilization: %.1f%%\n", utilization);
// 目标：> 80% 对于简单的 element-wise 操作
```

---

## 总结

### Element-wise 算子的核心要点：

1. **本质是 Memory-bound**：优化重点是减少内存访问
2. **融合是关键**：多个操作融合 > 单个操作优化
3. **向量化必须**：使用 float4 等向量类型
4. **快速数学**：使用内置快速函数
5. **简单即最好**：对于 element-wise，简单的优化通常最有效

### 优化优先级：

```
1. 算子融合 (3-10x)
   ↓
2. 向量化 (1.5-2x)
   ↓
3. 快速数学函数 (1.5-3x for 超越函数)
   ↓
4. 其他细节优化 (1.1-1.3x)
```

### 最终目标：

**达到理论内存带宽的 80-90%**

```python
# 检查是否达到目标
bandwidth_utilization = actual_bw / theoretical_bw
if bandwidth_utilization > 0.8:
    print("✅ Excellent! Close to peak memory bandwidth")
elif bandwidth_utilization > 0.6:
    print("⚠️  Good, but room for improvement")
else:
    print("❌ Need optimization")
```
