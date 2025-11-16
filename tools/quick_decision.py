#!/usr/bin/env python3
"""
快速决策工具 - 一键诊断 GPU Kernel 瓶颈

使用:
    python quick_decision.py <binary> [args...]

示例:
    python quick_decision.py ./vector_add
    python quick_decision.py ./gemm 1024 1024 1024
"""

import subprocess
import sys
import os
from dataclasses import dataclass


@dataclass
class QuickDiagnosis:
    """快速诊断结果"""
    bottleneck: str  # Memory-bound / Compute-bound / Launch-bound
    problem: str     # 具体问题
    solution: str    # 优化方案
    expected_speedup: str  # 预期提升
    effort: str      # 工作量
    roi: str         # ROI 评级


def run_ncu_quick(binary, args=None):
    """运行 NCU 采集关键指标"""

    print("=" * 70)
    print("🔍 正在运行 NCU 快速诊断...")
    print("=" * 70)

    # 只采集 5 个关键指标 (快速模式)
    metrics = [
        "gpu__time_duration.avg",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory.avg.pct_of_peak_sustained_elapsed",
        "l1tex__average_t_sectors_per_request",
        "dram__bytes.sum",
    ]

    cmd = [
        "ncu",
        "--metrics", ",".join(metrics),
        "--csv",
        binary
    ]

    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 分钟超时
        )

        return result.stdout

    except subprocess.TimeoutExpired:
        print("❌ NCU 运行超时 (> 2 分钟)")
        print("   建议: 减小问题规模或使用更快的 baseline")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 找不到 ncu 命令")
        print("   请确保 CUDA Toolkit 已安装并在 PATH 中")
        sys.exit(1)


def parse_ncu_output(output):
    """解析 NCU CSV 输出"""

    metrics = {}

    lines = output.strip().split('\n')

    for line in lines:
        parts = line.split(',')

        if len(parts) < 5:
            continue

        # 移除引号
        parts = [p.strip('"') for p in parts]

        # CSV 格式: "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
        if parts[2] in [
            "gpu__time_duration.avg",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory.avg.pct_of_peak_sustained_elapsed",
            "l1tex__average_t_sectors_per_request",
            "dram__bytes.sum"
        ]:
            metric_name = parts[2]
            metric_value = parts[4] if len(parts) > 4 else parts[3]

            # 移除百分号
            metric_value = metric_value.replace('%', '').replace(',', '')

            try:
                metrics[metric_name] = float(metric_value)
            except:
                pass

    return metrics


def diagnose(metrics):
    """快速诊断瓶颈"""

    duration_ns = metrics.get("gpu__time_duration.avg", 0)
    sm_pct = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
    mem_pct = metrics.get("gpu__compute_memory.avg.pct_of_peak_sustained_elapsed", 0)
    sectors = metrics.get("l1tex__average_t_sectors_per_request", 0)
    dram_bytes = metrics.get("dram__bytes.sum", 0)

    duration_ms = duration_ns / 1e6

    # 诊断逻辑
    if mem_pct > 60 and sm_pct < 40:
        # Memory-bound
        if sectors > 1.5:
            return QuickDiagnosis(
                bottleneck="Memory-bound (显存瓶颈)",
                problem=f"非合并访问 (sectors_per_request={sectors:.2f})\n" +
                        f"       带宽浪费: {(sectors - 1) / sectors * 100:.1f}%",
                solution="修复 Memory Coalescing\n" +
                         "       → 确保相邻线程访问相邻内存\n" +
                         "       → 代码模式: data[tid] 而非 data[tid * stride]",
                expected_speedup="8-10x",
                effort="30 分钟",
                roi="⭐⭐⭐⭐⭐ 极高 ROI，立即执行"
            )
        else:
            # 访问已合并，但仍然 memory-bound
            if dram_bytes > 0:
                return QuickDiagnosis(
                    bottleneck="Memory-bound (显存瓶颈)",
                    problem=f"访问已合并 (sectors={sectors:.2f}) ✅\n" +
                            f"       但显存流量高: {dram_bytes / 1e9:.2f} GB\n" +
                            f"       → 可能存在重复读取",
                    solution="Shared Memory Tiling (如果有数据重用)\n" +
                             "       → 适用于: GEMM, Convolution, Reduction\n" +
                             "       → 不适用于: Element-wise 操作 (无重用)",
                    expected_speedup="10-20x (如果有重用)",
                    effort="2-4 小时",
                    roi="⭐⭐⭐⭐ 高 ROI (如果适用)"
                )
            else:
                return QuickDiagnosis(
                    bottleneck="Memory-bound (显存瓶颈)",
                    problem=f"访问已优化 (sectors={sectors:.2f})",
                    solution="已接近显存带宽极限\n" +
                             "       → 考虑 Vectorization (float4)\n" +
                             "       → 或转向算子融合",
                    expected_speedup="1.5-2x (vectorization)",
                    effort="1 小时",
                    roi="⭐⭐⭐ 中等 ROI"
                )

    elif sm_pct > 60 and mem_pct < 40:
        # Compute-bound
        return QuickDiagnosis(
            bottleneck="Compute-bound (计算瓶颈)",
            problem=f"计算单元利用率高 (SM {sm_pct:.1f}%)\n" +
                    f"       显存不是瓶颈 (Memory {mem_pct:.1f}%)",
            solution="使用 Tensor Core 加速\n" +
                     "       → 最简单: cuBLAS / cuDNN\n" +
                     "       → 手写: WMMA API (FP16/TF32)\n" +
                     "       → 预期: 10-16x 提升",
            expected_speedup="10-16x",
            effort="5 分钟 (cuBLAS) 或 1-2 天 (WMMA)",
            roi="⭐⭐⭐⭐⭐ (cuBLAS) 或 ⭐⭐ (手写)"
        )

    elif sm_pct < 40 and mem_pct < 40:
        # Launch-bound
        return QuickDiagnosis(
            bottleneck="Launch-bound (并行度不足)",
            problem=f"SM 和 Memory 都利用率低\n" +
                    f"       SM: {sm_pct:.1f}%, Memory: {mem_pct:.1f}%",
            solution="增加并行度\n" +
                     "       → 增加 block 数量\n" +
                     "       → 增加 threads per block\n" +
                     "       → 检查问题规模是否太小",
            expected_speedup="2-5x",
            effort="30 分钟",
            roi="⭐⭐⭐⭐"
        )

    else:
        # Balanced
        efficiency = max(sm_pct, mem_pct)
        if efficiency > 80:
            return QuickDiagnosis(
                bottleneck="已优化 (Balanced)",
                problem=f"SM: {sm_pct:.1f}%, Memory: {mem_pct:.1f}%\n" +
                        f"       效率 {efficiency:.1f}% > 80%",
                solution="Kernel 已达到较好效率\n" +
                         "       → 继续优化 ROI 很低\n" +
                         "       → 建议转向:\n" +
                         "         • 算子融合 (减少 kernel launch)\n" +
                         "         • 使用官方库 (cuBLAS/cuDNN)\n" +
                         "         • 系统级优化 (多 GPU, Pipeline)",
                expected_speedup="1.2-1.5x (收益递减)",
                effort="3+ 天",
                roi="❌ 低 ROI，不建议"
            )
        else:
            return QuickDiagnosis(
                bottleneck="混合瓶颈",
                problem=f"SM: {sm_pct:.1f}%, Memory: {mem_pct:.1f}%",
                solution="需要更详细的 NCU 分析\n" +
                         "       运行: ncu --set full --export report ./kernel\n" +
                         "       查看: NCU_STEP_BY_STEP.md",
                expected_speedup="未知",
                effort="需要详细分析",
                roi="⭐⭐⭐"
            )


def print_diagnosis(diag, duration_ms):
    """打印诊断结果"""

    print("\n" + "=" * 70)
    print("📊 诊断结果")
    print("=" * 70)
    print()

    print(f"⏱️  当前性能: {duration_ms:.4f} ms")
    print()

    print(f"🎯 瓶颈类型: {diag.bottleneck}")
    print()

    print(f"❌ 具体问题:")
    for line in diag.problem.split('\n'):
        print(f"   {line}")
    print()

    print(f"✅ 优化方案:")
    for line in diag.solution.split('\n'):
        print(f"   {line}")
    print()

    print(f"📈 预期提升: {diag.expected_speedup}")
    print(f"⏳ 工作量: {diag.effort}")
    print(f"💰 ROI 评级: {diag.roi}")
    print()

    print("=" * 70)
    print()

    # 给出下一步建议
    print("📋 下一步:")
    if "Coalescing" in diag.solution:
        print("   1. 查看代码中的内存访问模式")
        print("   2. 确保 data[tid] 而非 data[tid * stride]")
        print("   3. 参考: benchmarks/vector_ops/vector_add/v0_naive.cu → v1_coalesced.cu")
    elif "Shared Memory" in diag.solution:
        print("   1. 确认是否有数据重用 (如矩阵乘法)")
        print("   2. 参考: benchmarks/matrix_ops/gemm/v2_shared_tiling.cu")
        print("   3. 注意: Element-wise 操作不需要 Shared Memory")
    elif "Tensor Core" in diag.solution:
        print("   1. 如果是矩阵乘法，直接用 cuBLAS")
        print("   2. cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)")
        print("   3. 参考: techniques/cuda_triton_optimization_techniques.md - Tensor Core 章节")
    elif "已优化" in diag.bottleneck:
        print("   1. 停止优化 kernel")
        print("   2. 考虑算子融合或系统级优化")
        print("   3. 查看: GPU_OPTIMIZATION_ESSENCE.md - 停止优化的信号")
    else:
        print("   1. 运行完整 NCU: ncu --set full --export report ./kernel")
        print("   2. 查看: NCU_STEP_BY_STEP.md")

    print()


def main():
    if len(sys.argv) < 2:
        print("用法: python quick_decision.py <binary> [args...]")
        print()
        print("示例:")
        print("  python quick_decision.py ./vector_add")
        print("  python quick_decision.py ./gemm 1024 1024 1024")
        sys.exit(1)

    binary = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else None

    if not os.path.exists(binary):
        print(f"❌ 文件不存在: {binary}")
        sys.exit(1)

    # 运行 NCU
    output = run_ncu_quick(binary, args)

    # 解析结果
    metrics = parse_ncu_output(output)

    if not metrics:
        print("❌ 无法解析 NCU 输出")
        print("   原始输出:")
        print(output)
        sys.exit(1)

    # 诊断
    diagnosis = diagnose(metrics)

    # 显示结果
    duration_ms = metrics.get("gpu__time_duration.avg", 0) / 1e6
    print_diagnosis(diagnosis, duration_ms)

    # 显示原始指标 (调试用)
    print("📊 原始 NCU 指标:")
    print(f"   Duration: {duration_ms:.4f} ms")
    print(f"   SM Throughput: {metrics.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0):.1f}%")
    print(f"   Memory Throughput: {metrics.get('gpu__compute_memory.avg.pct_of_peak_sustained_elapsed', 0):.1f}%")
    print(f"   Sectors/Request: {metrics.get('l1tex__average_t_sectors_per_request', 0):.2f}")
    print(f"   DRAM Bytes: {metrics.get('dram__bytes.sum', 0) / 1e9:.2f} GB")
    print()


if __name__ == "__main__":
    main()
