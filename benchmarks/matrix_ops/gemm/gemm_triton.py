"""
GEMM - Triton Implementation

Triton 自动优化:
- 自动 tiling 和 blocking
- 自动 shared memory 管理
- 自动 bank conflict 避免
- 自动 loop unrolling

性能: 接近 CUDA v3_optimized，代码量 1/10
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton GEMM Kernel

    C = A × B
    A: M × K
    B: K × N
    C: M × N
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Determine which block of C this program computes
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Iterate to compute a block of the C matrix
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Matrix multiply accumulate
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def gemm_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton GEMM wrapper

    Args:
        a: [M, K] matrix
        b: [K, N] matrix

    Returns:
        c: [M, N] matrix
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Tunable parameters
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch kernel
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return c


def benchmark_gemm(M=1024, N=1024, K=1024, num_iterations=10):
    """Benchmark Triton GEMM"""
    print(f"=== GEMM Triton Implementation ===")
    print(f"Matrix sizes: A({M}×{K}) × B({K}×{N}) = C({M}×{N})")
    print("")

    # Create tensors
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')

    # Warmup
    c = gemm_triton(a, b)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        c = gemm_triton(a, b)
    end_event.record()

    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / num_iterations

    # Verify
    c_ref = torch.matmul(a, b)
    assert torch.allclose(c, c_ref, rtol=1e-3, atol=1e-3), "Verification failed!"

    # Calculate GFLOPS
    gflops = (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e9

    print(f"=== Performance ===")
    print(f"Time: {avg_ms:.4f} ms")
    print(f"Performance: {gflops:.2f} GFLOPS")
    print(f"Efficiency: {(gflops / 19500.0) * 100:.2f}% (vs A100 FP32 peak)")
    print(f"Verification: PASSED")

    print(f"\n=== CUDA vs Triton Comparison ===")
    print(f"CUDA v0 (Naive):      ~150 GFLOPS   (~100 lines code)")
    print(f"CUDA v2 (Shared):     ~2500 GFLOPS  (~150 lines code)")
    print(f"CUDA v3 (Optimized):  ~5000 GFLOPS  (~180 lines code)")
    print(f"Triton:               ~{gflops:.0f} GFLOPS  (~60 lines code) ✅")
    print(f"\nTriton 优势:")
    print(f"  ✅ 自动 tiling 和 blocking")
    print(f"  ✅ 自动 shared memory 管理")
    print(f"  ✅ 自动 bank conflict 避免")
    print(f"  ✅ 代码量 1/3")
    print(f"\nCUDA 优势:")
    print(f"  ✅ Tensor Core 支持更好 (WMMA, cuBLAS)")
    print(f"  ✅ 极致优化空间更大 (可达 10-15 TFLOPS FP32)")

    return avg_ms, gflops


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # Run benchmark
    benchmark_gemm(1024, 1024, 1024)

    print(f"\n=== 不同矩阵尺寸性能 ===")
    for size in [512, 1024, 2048, 4096]:
        avg_ms, gflops = benchmark_gemm(size, size, size)
        print(f"{size}×{size}×{size}: {gflops:.0f} GFLOPS ({avg_ms:.2f} ms)")
