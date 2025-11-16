# Triton 算子编写完整教程

## 目录
1. [Triton 简介](#triton-简介)
2. [第一个 Triton Kernel](#第一个-triton-kernel)
3. [Triton 编程模型](#triton-编程模型)
4. [自动调优](#自动调优)
5. [完整示例](#完整示例)
6. [Triton vs CUDA](#triton-vs-cuda)

---

## Triton 简介

### 什么是 Triton？

Triton 是 OpenAI 开发的一种 GPU 编程语言，旨在简化高性能 GPU kernel 的编写。

**核心优势：**
- 基于 Python，易于学习和使用
- 自动内存管理和优化
- 自动调优（Auto-tuning）
- Block-level 编程，抽象级别高于 CUDA
- JIT 编译，性能接近手写 CUDA

**适用场景：**
- 快速原型开发
- 算子性能优化
- 深度学习自定义算子
- 研究新算法

---

## 第一个 Triton Kernel

### 向量加法

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,  # 输入指针 X
    y_ptr,  # 输入指针 Y
    output_ptr,  # 输出指针
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    # 获取当前 program 的 ID
    pid = tl.program_id(axis=0)

    # 计算当前 block 处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建 mask 处理边界情况
    mask = offsets < n_elements

    # 加载数据（使用 mask 避免越界）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 计算
    output = x + y

    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor):
    # 分配输出张量
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()

    # 配置 grid：需要多少个 block
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 启动 kernel
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


# 使用示例
if __name__ == "__main__":
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    output_triton = vector_add(x, y)
    output_torch = x + y

    print(f'Triton vs PyTorch: {torch.allclose(output_triton, output_torch)}')
```

### 关键概念解析

#### 1. `@triton.jit` 装饰器
- 标记函数为 Triton kernel
- 启用 JIT 编译
- 进行自动优化

#### 2. `tl.program_id(axis=0)`
- 类似 CUDA 的 `blockIdx.x`
- 获取当前 program（block）的 ID

#### 3. `tl.constexpr`
- 编译时常量
- 用于优化代码生成
- 可以作为模板参数

#### 4. `tl.load` 和 `tl.store`
- 向量化加载/存储
- 自动处理内存合并访问
- 支持 mask 操作

#### 5. Grid 配置
```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
```
- 动态计算需要的 block 数量
- `meta` 包含编译时参数

---

## Triton 编程模型

### Block-level 编程

Triton 采用 Block-level 编程模型，每个 program 处理一个数据块。

```
┌─────────────────────────────┐
│        Input Data           │
└─────────────────────────────┘
       ↓        ↓        ↓
   [Block 0][Block 1][Block 2] ... (Programs)
       ↓        ↓        ↓
┌─────────────────────────────┐
│       Output Data           │
└─────────────────────────────┘
```

### 内存访问模式

#### 一维数据访问
```python
@triton.jit
def kernel_1d(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(data_ptr + offsets, mask=mask)
```

#### 二维数据访问（矩阵）
```python
@triton.jit
def kernel_2d(data_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 行偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 列偏移
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 二维偏移
    offs = offs_m[:, None] * N + offs_n[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    data = tl.load(data_ptr + offs, mask=mask)
```

### 数据类型

```python
# 标量类型
tl.float16, tl.float32, tl.float64
tl.int8, tl.int16, tl.int32, tl.int64

# 向量操作
x = tl.load(ptr + offsets)  # 加载向量
y = x * 2.0                  # 向量运算
z = tl.sum(y)                # 向量规约

# 类型转换
x_fp16 = x.to(tl.float16)
```

---

## 自动调优

Triton 的杀手级特性：自动调优（Auto-tuning）

### 基础自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def vector_add_autotuned(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # kernel 实现
    pass
```

### 高级自动调优

```python
def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                     num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=5, num_warps=2),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # kernel 实现
    pass
```

**参数说明：**
- `num_warps`：每个 block 的 warp 数量
- `num_stages`：软件流水线阶段数
- `key`：用于缓存最佳配置的键

---

## 完整示例：矩阵乘法

### Triton 实现

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8},
                     num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                     num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # 指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # stride（用于处理非连续张量）
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Kernel for matrix multiplication C = A @ B
    A: (M, K)
    B: (K, N)
    C: (M, N)
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 创建指针偏移
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 迭代计算
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 tile
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # 矩阵乘法
        accumulator += tl.dot(a, b)

        # 更新指针
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    # 写回结果
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    """矩阵乘法封装函数"""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape

    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 配置 grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    # 启动 kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        GROUP_M=8,
    )

    return c


# 性能测试
if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)

    # Triton
    triton_output = matmul(a, b)

    # PyTorch
    torch_output = torch.matmul(a, b)

    print(f"Triton vs PyTorch: {torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)}")

    # Benchmark
    import time

    # Warmup
    for _ in range(10):
        _ = matmul(a, b)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    triton_time = (end - start) / 100 * 1000
    print(f"Triton time: {triton_time:.3f} ms")
```

---

## Triton vs CUDA

### 对比表

| 特性 | CUDA | Triton |
|-----|------|--------|
| 语言 | C++ | Python |
| 学习曲线 | 陡峭 | 平缓 |
| 开发效率 | 低 | 高 |
| 性能 | 最优 | 接近CUDA (80-100%) |
| 内存管理 | 手动 | 自动 |
| 优化难度 | 高 | 低（自动调优） |
| 调试 | 困难 | 相对容易 |
| 可移植性 | GPU特定 | 更好 |

### 代码对比：向量加法

#### CUDA
```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 主机代码
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, bytes);
cudaMalloc(&d_B, bytes);
cudaMalloc(&d_C, bytes);
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
int threads = 256;
int blocks = (N + threads - 1) / threads;
vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
```

#### Triton
```python
@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

# 使用
output = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
vector_add[grid](x, y, output, n, BLOCK=1024)
```

### 何时使用 CUDA？

- 需要极致性能（每一点性能都很关键）
- 需要精细控制硬件资源
- 实现非常复杂的算法
- 特定硬件优化（Tensor Cores, etc.）

### 何时使用 Triton？

- 快速原型开发
- 研究新算法
- 大多数深度学习算子
- 需要快速迭代的场景
- 团队 Python 背景强于 C++

---

## 高级特性

### 1. Atomic 操作

```python
@triton.jit
def atomic_add_kernel(data_ptr, indices_ptr, values_ptr, n):
    pid = tl.program_id(0)
    idx = tl.load(indices_ptr + pid)
    val = tl.load(values_ptr + pid)
    tl.atomic_add(data_ptr + idx, val)
```

### 2. 规约操作

```python
@triton.jit
def sum_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Block 内规约
    sum_val = tl.sum(x)

    # 存储结果
    if pid == 0:
        tl.store(output_ptr, sum_val)
```

### 3. Softmax

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # 加载数据
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    # 数值稳定的 softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)
```

---

## 调试技巧

### 1. 打印调试
```python
@triton.jit
def debug_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)

    # 只在第一个 block 打印
    if pid == 0:
        tl.device_print("pid:", pid)
        tl.device_print("BLOCK:", BLOCK)
```

### 2. 检查数值正确性
```python
# 与 PyTorch 实现对比
torch_result = torch.softmax(x, dim=-1)
triton_result = softmax_triton(x)
print(f"Max diff: {(torch_result - triton_result).abs().max()}")
print(f"Allclose: {torch.allclose(torch_result, triton_result, atol=1e-2)}")
```

### 3. 性能分析
```python
# 使用 Triton 内置的 benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        ylabel='GB/s',
        plot_name='vector-add-performance',
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add(x, y), quantiles=quantiles)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)
```

---

## 最佳实践

### 1. 选择合适的 Block 大小
- 通常从 256、512、1024 开始尝试
- 使用 autotune 找到最优配置
- 考虑数据对齐

### 2. 内存访问优化
- 尽量连续访问内存
- 利用 mask 避免分支
- 预取数据（使用 num_stages）

### 3. 利用自动调优
- 为不同输入大小配置不同参数
- 使用 key 参数缓存最优配置
- 测试多种 num_warps 和 num_stages 组合

### 4. 数值稳定性
- Softmax 等操作注意数值稳定性
- 使用合适的数据类型（fp16, fp32）
- 注意累加精度

---

## 下一步

- 查看 [examples](examples/) 目录中的完整代码
- 学习 [性能优化指南](../03_performance_optimization/optimization_guide.md)
- 对比 CUDA 和 Triton 的实现差异
- 尝试实现自己的算子

## 参考资料

- [Triton 官方文档](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton 教程集合](https://triton-lang.org/main/getting-started/tutorials/index.html)
