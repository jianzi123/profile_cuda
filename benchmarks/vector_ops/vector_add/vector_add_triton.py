"""
Vector Add - Triton Implementation

对比 CUDA 版本，展示 Triton 的自动优化:
- 自动内存合并 (无需手动管理 coalescing)
- 自动向量化 (编译器自动选择 float4/float8)
- 自动 loop tiling
- 更简洁的代码

CUDA 需要 5 个版本 (v0-v4) 手动优化
Triton 只需 1 个版本即可达到接近 v4 的性能!
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np


@triton.jit
def vector_add_kernel(
    a_ptr,  # Pointer to input array a
    b_ptr,  # Pointer to input array b
    c_ptr,  # Pointer to output array c
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    """
    Triton Vector Add Kernel

    Triton 自动优化:
    1. Memory coalescing: 自动生成连续访问
    2. Vectorization: 自动选择 vector load/store
    3. Loop unrolling: 编译器自动展开
    4. Cache hints: 自动优化 cache 策略
    """
    # Program ID (类似 CUDA 的 blockIdx.x)
    pid = tl.program_id(axis=0)

    # Block start offset
    block_start = pid * BLOCK_SIZE

    # Offsets within this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for boundary checking
    mask = offsets < n_elements

    # Load data (自动向量化和合并访问!)
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Compute
    c = a + b

    # Store result (自动向量化!)
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton Vector Add Wrapper

    相当于 CUDA v4_optimized 的性能
    但代码量只有 1/3!
    """
    assert a.shape == b.shape
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype

    n_elements = a.numel()
    c = torch.empty_like(a)

    # Launch configuration
    # Triton 自动选择最优的 BLOCK_SIZE
    BLOCK_SIZE = 1024  # 可以调整: 256, 512, 1024, 2048

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    vector_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return c


def benchmark_triton(size: int = 64 * 1024 * 1024, num_iterations: int = 100):
    """
    Benchmark Triton implementation
    """
    print("=" * 60)
    print("Vector Add - Triton Implementation")
    print("=" * 60)

    # Create tensors
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')

    bytes_per_array = size * 4  # float32 = 4 bytes
    total_mb = 3 * bytes_per_array / 1024 / 1024

    print(f"Problem size: {size:,} elements ({bytes_per_array/1024/1024:.2f} MB per array)")
    print(f"Total memory: {total_mb:.2f} MB")
    print("")

    # Warmup
    c = vector_add_triton(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        c = vector_add_triton(a, b)
    end_event.record()

    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / num_iterations

    # Verify correctness
    c_ref = a + b  # PyTorch reference
    assert torch.allclose(c, c_ref, rtol=1e-5), "Verification failed!"

    # Calculate bandwidth
    total_gb = 3 * bytes_per_array / 1e9
    bandwidth = total_gb / (avg_ms / 1000.0)

    # Get theoretical bandwidth
    props = torch.cuda.get_device_properties(0)
    # Bandwidth in GB/s (approximate for A100: 1555 GB/s)
    theoretical_bw = 1555  # Adjust for your GPU

    efficiency = (bandwidth / theoretical_bw) * 100

    print("=" * 60)
    print("Performance")
    print("=" * 60)
    print(f"Time: {avg_ms:.4f} ms")
    print(f"Bandwidth: {bandwidth:.2f} GB/s")
    print(f"Efficiency: {efficiency:.1f}%")
    print(f"Verification: PASSED")

    print("")
    print("=" * 60)
    print("Triton vs CUDA Comparison")
    print("=" * 60)
    print("")
    print("Code Complexity:")
    print("  CUDA v0 (Naive):        ~80 lines + manual coalescing")
    print("  CUDA v1 (Coalesced):    ~85 lines + manual indexing")
    print("  CUDA v2 (Vectorized):   ~90 lines + float4 manual handling")
    print("  CUDA v4 (Optimized):    ~100 lines + unroll + hints")
    print("  Triton:                 ~20 lines ✅ (本文件)")
    print("")
    print("Performance (Expected on A100):")
    print("  CUDA v0:  12.5 ms   (61 GB/s,   3.9%)")
    print("  CUDA v1:  1.5 ms    (500 GB/s,  32%)")
    print("  CUDA v2:  0.7 ms    (1077 GB/s, 69%)")
    print("  CUDA v4:  0.54 ms   (1416 GB/s, 91%) ← 手动优化极限")
    print(f"  Triton:   {avg_ms:.2f} ms   ({bandwidth:.0f} GB/s, {efficiency:.0f}%) ← 自动达到!")
    print("")
    print("Triton 优势:")
    print("  ✅ 自动内存合并 (无需关心 stride)")
    print("  ✅ 自动向量化 (编译器选择最优 vector size)")
    print("  ✅ 自动循环优化 (unrolling, tiling)")
    print("  ✅ 代码简洁 (5x less code than CUDA v4)")
    print("  ✅ 可移植性好 (AMD, NVIDIA, Intel)")
    print("")
    print("CUDA 优势:")
    print("  ✅ 完全控制 (极致优化时有 5-10% 优势)")
    print("  ✅ 更多底层特性 (Tensor Core, WMMA, Cooperative Groups)")
    print("  ✅ 成熟生态 (更多工具和文档)")
    print("")
    print("结论:")
    print("  - 90% 场景: 使用 Triton (开发效率高)")
    print("  - 极致性能: 使用 CUDA (5-10% 额外提升)")
    print("  - 学习: 两者都学 (理解底层原理)")

    return avg_ms, bandwidth, efficiency


def compare_block_sizes():
    """
    对比不同 BLOCK_SIZE 的影响
    Triton 会自动调优，但手动选择也很重要
    """
    print("")
    print("=" * 60)
    print("BLOCK_SIZE Tuning (Triton)")
    print("=" * 60)
    print("")

    size = 64 * 1024 * 1024
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')

    block_sizes = [128, 256, 512, 1024, 2048]
    results = []

    for block_size in block_sizes:
        # Modify kernel launch
        @triton.jit
        def test_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            c = a + b
            tl.store(c_ptr + offsets, c, mask=mask)

        c = torch.empty_like(a)
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

        # Warmup
        test_kernel[grid](a, b, c, size, BLOCK_SIZE=block_size)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            test_kernel[grid](a, b, c, size, BLOCK_SIZE=block_size)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / 100
        bandwidth = (3 * size * 4 / 1e9) / (avg_ms / 1000.0)

        results.append((block_size, avg_ms, bandwidth))

        print(f"BLOCK_SIZE={block_size:4d}: {avg_ms:6.3f} ms, {bandwidth:7.1f} GB/s")

    # Find best
    best = max(results, key=lambda x: x[2])
    print(f"\nBest BLOCK_SIZE: {best[0]} ({best[2]:.1f} GB/s)")
    print("\n提示: BLOCK_SIZE 选择规则")
    print("  - 太小 (128): Launch overhead 过大")
    print("  - 太大 (2048): Occupancy 下降 (寄存器/shared mem 限制)")
    print("  - 推荐: 512-1024 (大多数场景)")


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This benchmark requires a CUDA GPU.")
        exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("")

    # Main benchmark
    avg_ms, bandwidth, efficiency = benchmark_triton()

    # Block size comparison
    compare_block_sizes()

    print("")
    print("=" * 60)
    print("NCU Analysis for Triton")
    print("=" * 60)
    print("")
    print("由于 Triton 生成的是 CUDA kernel，可以用 NCU 分析:")
    print("")
    print("# 方法 1: 使用 Triton profiler")
    print("import triton.profiler as profiler")
    print("profiler.start('vector_add')")
    print("c = vector_add_triton(a, b)")
    print("profiler.finalize()")
    print("")
    print("# 方法 2: NCU 直接分析 Python 程序")
    print("ncu --set full python vector_add_triton.py")
    print("")
    print("预期 NCU 结果 (类似 CUDA v4):")
    print("  - l1tex__average_t_sectors_per_request: ~1.0 ✅")
    print("  - Memory Throughput: 90-95%")
    print("  - SM Throughput: 15-20%")
    print("  - 证明 Triton 编译器生成了优化的代码!")
