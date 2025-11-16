#!/usr/bin/env python3
"""
Triton implementation of vector addition
Demonstrates basic Triton kernel and auto-tuning
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def vector_add_kernel(
    x_ptr,  # 输入指针 X
    y_ptr,  # 输入指针 Y
    output_ptr,  # 输出指针
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 块大小（编译时常量）
):
    """
    向量加法 kernel
    C = A + B
    """
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
def vector_add_kernel_autotuned(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """自动调优版本"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add_triton(x: torch.Tensor, y: torch.Tensor, autotuned=False):
    """
    Triton 向量加法封装函数
    """
    # 分配输出张量
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.shape == y.shape

    n_elements = output.numel()

    # 配置 grid：需要多少个 block
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 启动 kernel
    if autotuned:
        vector_add_kernel_autotuned[grid](x, y, output, n_elements)
    else:
        vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


def benchmark(fn, x, y, name="", num_iterations=100):
    """性能测试函数"""
    # Warmup
    for _ in range(10):
        _ = fn(x, y)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        output = fn(x, y)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_iterations * 1000  # ms

    # 计算带宽
    bytes_accessed = 3 * x.numel() * x.element_size()  # Read x, Read y, Write output
    bandwidth = (bytes_accessed / 1e9) / (avg_time / 1000)  # GB/s

    print(f"{name:30s}: {avg_time:.3f} ms, {bandwidth:.2f} GB/s")

    return output, avg_time


def main():
    print("=" * 70)
    print("Triton Vector Addition Benchmark")
    print("=" * 70)

    # 测试不同大小
    sizes = [1000000, 10000000, 100000000]  # 1M, 10M, 100M

    for size in sizes:
        print(f"\nVector size: {size:,}")
        print("-" * 70)

        # 创建输入数据
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)

        # PyTorch baseline
        output_torch, torch_time = benchmark(
            lambda a, b: a + b, x, y,
            name="PyTorch",
            num_iterations=100
        )

        # Triton (固定 block size)
        output_triton, triton_time = benchmark(
            lambda a, b: vector_add_triton(a, b, autotuned=False),
            x, y,
            name="Triton (BLOCK_SIZE=1024)",
            num_iterations=100
        )

        # Triton (auto-tuned)
        output_triton_tuned, triton_tuned_time = benchmark(
            lambda a, b: vector_add_triton(a, b, autotuned=True),
            x, y,
            name="Triton (Auto-tuned)",
            num_iterations=100
        )

        # 验证正确性
        if torch.allclose(output_triton, output_torch, rtol=1e-5):
            print("✓ Triton results verified!")
        else:
            print("✗ Triton results incorrect!")

        if torch.allclose(output_triton_tuned, output_torch, rtol=1e-5):
            print("✓ Triton auto-tuned results verified!")
        else:
            print("✗ Triton auto-tuned results incorrect!")

        # 性能对比
        print(f"\nSpeedup vs PyTorch:")
        print(f"  Triton (fixed):      {torch_time / triton_time:.2f}x")
        print(f"  Triton (auto-tuned): {torch_time / triton_tuned_time:.2f}x")


if __name__ == "__main__":
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    main()
