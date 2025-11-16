#!/usr/bin/env python3
"""
Automated NCU Profiling Tool

功能:
1. 自动运行 NCU 并收集关键指标
2. 解析 NCU 输出并提取性能数据
3. 自动诊断瓶颈 (Memory-bound/Compute-bound)
4. 生成优化建议

使用:
    python auto_profile.py <binary> [args]
    python auto_profile.py ./v0_naive
    python auto_profile.py ./gemm_v2 1024 1024 1024
"""

import subprocess
import json
import re
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Bottleneck(Enum):
    """瓶颈类型"""
    MEMORY_BOUND = "Memory-bound"
    COMPUTE_BOUND = "Compute-bound"
    LAUNCH_BOUND = "Launch-bound"
    BALANCED = "Balanced"
    UNKNOWN = "Unknown"


@dataclass
class ProfileResult:
    """NCU 分析结果"""
    binary: str
    kernel_name: str
    duration_ms: float
    sm_throughput: float  # Percentage
    memory_throughput: float  # Percentage
    sectors_per_request: float
    l2_hit_rate: float  # Percentage
    achieved_occupancy: float  # Percentage
    dram_bandwidth_gb_s: float
    bottleneck: Bottleneck
    optimization_suggestions: List[str]

    def __str__(self):
        return f"""
{'=' * 70}
NCU Profile Result
{'=' * 70}

Binary: {self.binary}
Kernel: {self.kernel_name}

Performance:
  Duration: {self.duration_ms:.4f} ms
  DRAM Bandwidth: {self.dram_bandwidth_gb_s:.2f} GB/s

Speed of Light:
  SM Throughput: {self.sm_throughput:.1f}% {'🔴' if self.sm_throughput < 40 else '🟡' if self.sm_throughput < 60 else '🟢'}
  Memory Throughput: {self.memory_throughput:.1f}% {'🔴' if self.memory_throughput < 40 else '🟡' if self.memory_throughput < 60 else '🟢'}

Memory Analysis:
  Sectors per Request: {self.sectors_per_request:.2f} {'✅' if self.sectors_per_request < 1.5 else '🔴'}
  L2 Hit Rate: {self.l2_hit_rate:.1f}%

Occupancy:
  Achieved: {self.achieved_occupancy:.1f}%

Bottleneck: {self.bottleneck.value}

Optimization Suggestions:
""" + '\n'.join(f"  • {s}" for s in self.optimization_suggestions)


class NCUProfiler:
    """NCU 自动化分析器"""

    # 关键指标列表
    KEY_METRICS = [
        "gpu__time_duration.avg",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory.avg.pct_of_peak_sustained_elapsed",
        "l1tex__average_t_sectors_per_request",
        "lts__t_sector_hit_rate.pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__bytes.sum",
        "smsp__sass_thread_inst_executed_op_fadd.sum",
        "smsp__sass_thread_inst_executed_op_fmul.sum",
        "smsp__sass_inst_executed_op_global_ld.sum",
        "smsp__average_warps_issue_stalled_long_scoreboard",
        "smsp__average_warps_issue_stalled_barrier",
    ]

    def __init__(self, ncu_path: str = "ncu"):
        """
        Args:
            ncu_path: NCU 可执行文件路径
        """
        self.ncu_path = ncu_path
        self._check_ncu_available()

    def _check_ncu_available(self):
        """检查 NCU 是否可用"""
        try:
            result = subprocess.run(
                [self.ncu_path, "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ NCU version: {result.stdout.strip()}")
        except FileNotFoundError:
            print(f"✗ Error: NCU not found at '{self.ncu_path}'")
            print("  Please install NVIDIA Nsight Compute or specify path")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running NCU: {e}")
            sys.exit(1)

    def profile(self, binary: str, args: List[str] = None) -> ProfileResult:
        """
        运行 NCU 并分析结果

        Args:
            binary: 要分析的二进制文件
            args: 传递给二进制文件的参数

        Returns:
            ProfileResult: 分析结果
        """
        if not os.path.exists(binary):
            raise FileNotFoundError(f"Binary not found: {binary}")

        print(f"\n{'=' * 70}")
        print(f"Profiling: {binary}")
        if args:
            print(f"Arguments: {' '.join(args)}")
        print(f"{'=' * 70}\n")

        # 构建 NCU 命令
        cmd = [
            self.ncu_path,
            "--metrics", ",".join(self.KEY_METRICS),
            "--csv",
            binary
        ]
        if args:
            cmd.extend(args)

        print(f"Running: {' '.join(cmd)}\n")

        # 运行 NCU
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit (NCU may return non-zero even on success)
            )

            # NCU 输出在 stdout 中
            output = result.stdout

            # 解析结果
            profile_result = self._parse_ncu_output(output, binary)

            return profile_result

        except subprocess.CalledProcessError as e:
            print(f"✗ Error running NCU:")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            raise

    def _parse_ncu_output(self, output: str, binary: str) -> ProfileResult:
        """解析 NCU CSV 输出"""

        # 分离 CSV 部分
        lines = output.strip().split('\n')

        # 找到 CSV 头
        csv_start = None
        for i, line in enumerate(lines):
            if '"ID"' in line or 'Metric Name' in line:
                csv_start = i
                break

        if csv_start is None:
            # 尝试找到包含指标的行
            for i, line in enumerate(lines):
                if 'gpu__time_duration' in line:
                    csv_start = i - 1  # Header 应该在前一行
                    break

        if csv_start is None:
            print("✗ Error: Could not parse NCU output")
            print("\nRaw output:")
            print(output)
            raise ValueError("Failed to parse NCU CSV output")

        # 解析 CSV
        csv_data = lines[csv_start:]

        # 提取指标值
        metrics = {}

        for line in csv_data:
            # CSV 格式: "ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
            parts = line.split(',')

            if len(parts) < 5:
                continue

            # 移除引号
            parts = [p.strip('"') for p in parts]

            if parts[2] in self.KEY_METRICS or any(m in parts[2] for m in self.KEY_METRICS):
                metric_name = parts[2]
                metric_value = parts[4] if len(parts) > 4 else parts[3]

                # 尝试转换为浮点数
                try:
                    # 处理百分号
                    if '%' in metric_value:
                        metric_value = metric_value.replace('%', '')
                    # 处理逗号分隔的大数字
                    metric_value = metric_value.replace(',', '')
                    metrics[metric_name] = float(metric_value)
                except ValueError:
                    # 保留字符串
                    metrics[metric_name] = metric_value

        # 提取 kernel 名称
        kernel_name = "unknown"
        for line in csv_data:
            parts = line.split(',')
            if len(parts) >= 2:
                parts = [p.strip('"') for p in parts]
                if parts[1] and parts[1] != "Kernel Name":
                    kernel_name = parts[1]
                    break

        # 计算派生指标
        duration_ms = metrics.get("gpu__time_duration.avg", 0) / 1e6  # ns to ms
        sm_throughput = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        memory_throughput = metrics.get("gpu__compute_memory.avg.pct_of_peak_sustained_elapsed", 0)
        sectors_per_request = metrics.get("l1tex__average_t_sectors_per_request", 0)
        l2_hit_rate = metrics.get("lts__t_sector_hit_rate.pct", 0)
        occupancy = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)

        # 计算 DRAM 带宽
        dram_bytes = metrics.get("dram__bytes.sum", 0)
        dram_bandwidth_gb_s = (dram_bytes / 1e9) / (duration_ms / 1000.0) if duration_ms > 0 else 0

        # 诊断瓶颈
        bottleneck = self._diagnose_bottleneck(
            sm_throughput,
            memory_throughput,
            sectors_per_request,
            occupancy
        )

        # 生成优化建议
        suggestions = self._generate_suggestions(
            bottleneck,
            sectors_per_request,
            l2_hit_rate,
            occupancy,
            metrics
        )

        return ProfileResult(
            binary=binary,
            kernel_name=kernel_name,
            duration_ms=duration_ms,
            sm_throughput=sm_throughput,
            memory_throughput=memory_throughput,
            sectors_per_request=sectors_per_request,
            l2_hit_rate=l2_hit_rate,
            achieved_occupancy=occupancy,
            dram_bandwidth_gb_s=dram_bandwidth_gb_s,
            bottleneck=bottleneck,
            optimization_suggestions=suggestions
        )

    def _diagnose_bottleneck(
        self,
        sm_throughput: float,
        memory_throughput: float,
        sectors_per_request: float,
        occupancy: float
    ) -> Bottleneck:
        """
        四象限诊断瓶颈

        参考 NCU UI Guide 中的 Speed of Light 分析
        """
        if memory_throughput > 60 and sm_throughput < 40:
            return Bottleneck.MEMORY_BOUND
        elif sm_throughput > 60 and memory_throughput < 40:
            return Bottleneck.COMPUTE_BOUND
        elif memory_throughput < 40 and sm_throughput < 40:
            if occupancy < 50:
                return Bottleneck.LAUNCH_BOUND
            else:
                return Bottleneck.UNKNOWN
        elif memory_throughput > 60 and sm_throughput > 60:
            return Bottleneck.BALANCED
        else:
            return Bottleneck.UNKNOWN

    def _generate_suggestions(
        self,
        bottleneck: Bottleneck,
        sectors_per_request: float,
        l2_hit_rate: float,
        occupancy: float,
        metrics: Dict[str, float]
    ) -> List[str]:
        """根据诊断结果生成优化建议"""

        suggestions = []

        # Memory coalescing
        if sectors_per_request > 1.5:
            suggestions.append(
                f"❌ 非合并访问 (sectors_per_request={sectors_per_request:.2f})"
                "\n     → 修复为连续访问 (预期 8-10x 提升)"
                "\n     → 参考: benchmarks/vector_ops/vector_add/v0→v1"
            )

        # Bottleneck specific
        if bottleneck == Bottleneck.MEMORY_BOUND:
            suggestions.append(
                "🔴 Memory-bound 瓶颈"
                "\n     优化方向:"
                "\n       1. Memory coalescing (如果 sectors_per_request > 1.5)"
                "\n       2. Vectorization (float4)"
                "\n       3. Shared Memory (如果有数据重用)"
                "\n       4. Kernel Fusion (减少显存往返)"
            )

            if sectors_per_request < 1.5:
                suggestions.append(
                    "✓ 访问已合并，考虑:"
                    "\n     → Vectorization (float4) - 1.5-2x 提升"
                    "\n     → Kernel Fusion - 2-5x 提升"
                )

        elif bottleneck == Bottleneck.COMPUTE_BOUND:
            suggestions.append(
                "🔴 Compute-bound 瓶颈"
                "\n     优化方向:"
                "\n       1. Tensor Core (FP16/TF32 for GEMM)"
                "\n       2. ILP (多个累加器)"
                "\n       3. Loop Unrolling"
                "\n       4. 算子融合 (减少 kernel launch)"
            )

        elif bottleneck == Bottleneck.LAUNCH_BOUND:
            suggestions.append(
                f"🔴 Launch-bound 瓶颈 (Occupancy={occupancy:.1f}%)"
                "\n     优化方向:"
                "\n       1. 增加 blocks/threads"
                "\n       2. 减少寄存器使用 (--maxrregcount)"
                "\n       3. 减少 Shared Memory 使用"
                "\n       4. 检查 warp divergence"
            )

        elif bottleneck == Bottleneck.BALANCED:
            suggestions.append(
                "✅ 性能已优化 (SM 和 Memory 都高利用率)"
                "\n     进一步优化方向:"
                "\n       1. 系统级优化 (kernel fusion, pipeline)"
                "\n       2. 多 GPU 并行"
                "\n       3. 混合精度 (FP16)"
            )

        # L2 cache
        if l2_hit_rate < 30 and bottleneck == Bottleneck.MEMORY_BOUND:
            suggestions.append(
                f"🟡 L2 Cache 命中率低 ({l2_hit_rate:.1f}%)"
                "\n     → 正常对于 streaming workload"
                "\n     → 如果有局部性，考虑 tiling/blocking"
            )

        # Occupancy
        if occupancy < 50:
            suggestions.append(
                f"🟡 Occupancy 低 ({occupancy:.1f}%)"
                "\n     可能原因:"
                "\n       • 寄存器使用过多"
                "\n       • Shared Memory 使用过多"
                "\n       • Block size 太大/太小"
                "\n     → 使用 --resource-usage 查看限制因素"
            )

        # Barrier stalls
        barrier_stall = metrics.get("smsp__average_warps_issue_stalled_barrier", 0)
        if barrier_stall > 20:
            suggestions.append(
                f"🟡 Barrier Stall 高 ({barrier_stall:.1f}%)"
                "\n     → __syncthreads() 过多或 warp divergence"
                "\n     → 检查是否误用 Shared Memory (参考 v3_shared_tiling 反例)"
            )

        if not suggestions:
            suggestions.append("✅ 未发现明显瓶颈，性能良好")

        return suggestions


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python auto_profile.py <binary> [args...]")
        print("\nExamples:")
        print("  python auto_profile.py ./v0_naive")
        print("  python auto_profile.py ./gemm 1024 1024 1024")
        sys.exit(1)

    binary = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else None

    profiler = NCUProfiler()
    result = profiler.profile(binary, args)

    print(result)

    # 保存结果到 JSON
    output_file = f"{os.path.basename(binary)}_profile.json"
    with open(output_file, 'w') as f:
        json.dump({
            'binary': result.binary,
            'kernel_name': result.kernel_name,
            'duration_ms': result.duration_ms,
            'sm_throughput': result.sm_throughput,
            'memory_throughput': result.memory_throughput,
            'sectors_per_request': result.sectors_per_request,
            'l2_hit_rate': result.l2_hit_rate,
            'achieved_occupancy': result.achieved_occupancy,
            'dram_bandwidth_gb_s': result.dram_bandwidth_gb_s,
            'bottleneck': result.bottleneck.value,
            'optimization_suggestions': result.optimization_suggestions
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
