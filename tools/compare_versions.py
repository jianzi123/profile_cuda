#!/usr/bin/env python3
"""
Kernel Version Comparison Tool

功能:
1. 对比多个 kernel 版本的性能
2. 计算相对加速比
3. 生成对比表格
4. 识别性能回归

使用:
    python compare_versions.py <binary1> <binary2> ... [--baseline=<binary>]
    python compare_versions.py v0_naive v1_coalesced v2_vectorized v4_optimized
    python compare_versions.py v*.cu --baseline=v0_naive
"""

import subprocess
import json
import sys
import os
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
from auto_profile import NCUProfiler, ProfileResult


@dataclass
class ComparisonResult:
    """版本对比结果"""
    name: str
    profile: ProfileResult
    speedup: float  # 相对 baseline
    bandwidth_ratio: float  # 相对 baseline
    regression: bool  # 是否性能回归


class VersionComparator:
    """版本对比工具"""

    def __init__(self, profiler: NCUProfiler = None):
        self.profiler = profiler or NCUProfiler()
        self.results: List[ComparisonResult] = []

    def compare(
        self,
        binaries: List[str],
        baseline: str = None,
        args: List[str] = None
    ) -> List[ComparisonResult]:
        """
        对比多个版本

        Args:
            binaries: 二进制文件列表
            baseline: 基准版本 (默认第一个)
            args: 传递给每个二进制的参数

        Returns:
            List[ComparisonResult]: 对比结果
        """
        if not binaries:
            raise ValueError("No binaries provided")

        # 确定 baseline
        if baseline is None:
            baseline = binaries[0]
        elif baseline not in binaries:
            raise ValueError(f"Baseline '{baseline}' not in binaries list")

        # Profile 所有版本
        profiles = {}
        for binary in binaries:
            print(f"\n{'=' * 70}")
            print(f"Profiling: {binary}")
            print(f"{'=' * 70}")

            profile = self.profiler.profile(binary, args)
            profiles[binary] = profile

        # 获取 baseline 性能
        baseline_profile = profiles[baseline]
        baseline_time = baseline_profile.duration_ms
        baseline_bandwidth = baseline_profile.dram_bandwidth_gb_s

        # 计算对比结果
        results = []
        for binary in binaries:
            profile = profiles[binary]

            # 计算加速比
            if baseline_time > 0:
                speedup = baseline_time / profile.duration_ms
            else:
                speedup = 1.0

            # 计算带宽比
            if baseline_bandwidth > 0:
                bandwidth_ratio = profile.dram_bandwidth_gb_s / baseline_bandwidth
            else:
                bandwidth_ratio = 1.0

            # 检测性能回归 (相比前一个版本变慢)
            regression = False
            if len(results) > 0:
                prev_profile = results[-1].profile
                if profile.duration_ms > prev_profile.duration_ms * 1.05:  # 5% 容忍度
                    regression = True

            results.append(ComparisonResult(
                name=os.path.basename(binary),
                profile=profile,
                speedup=speedup,
                bandwidth_ratio=bandwidth_ratio,
                regression=regression
            ))

        self.results = results
        return results

    def print_comparison_table(self, baseline_name: str = None):
        """打印对比表格"""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "=" * 140)
        print("Performance Comparison Table")
        print("=" * 140)

        # 表头
        header = f"{'Version':<25} {'Time (ms)':>12} {'Speedup':>10} {'BW (GB/s)':>12} {'BW Ratio':>10} {'SM %':>8} {'Mem %':>8} {'SPR':>8} {'Bottleneck':<15} {'Status':<10}"
        print(header)
        print("-" * 140)

        # 数据行
        for i, result in enumerate(self.results):
            p = result.profile

            # 状态标记
            status = ""
            if i == 0:
                status = "Baseline"
            elif result.regression:
                status = "⚠️ Slower"
            elif result.speedup > self.results[i-1].speedup * 1.1:
                status = "✅ Faster"
            else:
                status = "➡️ Similar"

            # 格式化输出
            row = f"{result.name:<25} " \
                  f"{p.duration_ms:>12.4f} " \
                  f"{result.speedup:>9.2f}x " \
                  f"{p.dram_bandwidth_gb_s:>12.2f} " \
                  f"{result.bandwidth_ratio:>9.2f}x " \
                  f"{p.sm_throughput:>7.1f}% " \
                  f"{p.memory_throughput:>7.1f}% " \
                  f"{p.sectors_per_request:>8.2f} " \
                  f"{p.bottleneck.value:<15} " \
                  f"{status:<10}"

            print(row)

        print("=" * 140)

        # 总结
        print("\nSummary:")
        fastest = min(self.results, key=lambda r: r.profile.duration_ms)
        slowest = max(self.results, key=lambda r: r.profile.duration_ms)
        total_speedup = slowest.profile.duration_ms / fastest.profile.duration_ms

        print(f"  Fastest: {fastest.name} ({fastest.profile.duration_ms:.4f} ms)")
        print(f"  Slowest: {slowest.name} ({slowest.profile.duration_ms:.4f} ms)")
        print(f"  Total improvement: {total_speedup:.2f}x")

        # 性能回归警告
        regressions = [r for r in self.results if r.regression]
        if regressions:
            print(f"\n⚠️  Warning: {len(regressions)} version(s) with performance regression:")
            for r in regressions:
                print(f"     - {r.name}")

    def print_detailed_comparison(self):
        """打印详细对比"""
        if not self.results:
            return

        print("\n" + "=" * 70)
        print("Detailed Comparison")
        print("=" * 70)

        for i, result in enumerate(self.results):
            print(f"\n{i+1}. {result.name}")
            print(f"   Time: {result.profile.duration_ms:.4f} ms (Speedup: {result.speedup:.2f}x)")
            print(f"   Bandwidth: {result.profile.dram_bandwidth_gb_s:.2f} GB/s")
            print(f"   Bottleneck: {result.profile.bottleneck.value}")
            print(f"   Key Metrics:")
            print(f"     - SM Throughput: {result.profile.sm_throughput:.1f}%")
            print(f"     - Memory Throughput: {result.profile.memory_throughput:.1f}%")
            print(f"     - Sectors per Request: {result.profile.sectors_per_request:.2f}")
            print(f"     - Occupancy: {result.profile.achieved_occupancy:.1f}%")

            if result.regression:
                print(f"   ⚠️  REGRESSION: Slower than previous version")

            if i > 0:
                prev = self.results[i-1]
                improvement = (prev.profile.duration_ms - result.profile.duration_ms) / prev.profile.duration_ms * 100
                print(f"   vs {prev.name}: {improvement:+.1f}%")

    def export_json(self, output_file: str):
        """导出 JSON 格式对比结果"""
        data = {
            'comparison': [
                {
                    'name': r.name,
                    'time_ms': r.profile.duration_ms,
                    'speedup': r.speedup,
                    'bandwidth_gb_s': r.profile.dram_bandwidth_gb_s,
                    'bandwidth_ratio': r.bandwidth_ratio,
                    'sm_throughput': r.profile.sm_throughput,
                    'memory_throughput': r.profile.memory_throughput,
                    'sectors_per_request': r.profile.sectors_per_request,
                    'l2_hit_rate': r.profile.l2_hit_rate,
                    'occupancy': r.profile.achieved_occupancy,
                    'bottleneck': r.profile.bottleneck.value,
                    'regression': r.regression
                }
                for r in self.results
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Comparison exported to: {output_file}")

    def export_csv(self, output_file: str):
        """导出 CSV 格式对比结果"""
        import csv

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Version', 'Time (ms)', 'Speedup', 'Bandwidth (GB/s)',
                'BW Ratio', 'SM %', 'Memory %', 'Sectors/Req',
                'L2 Hit %', 'Occupancy %', 'Bottleneck', 'Regression'
            ])

            # Data
            for r in self.results:
                p = r.profile
                writer.writerow([
                    r.name,
                    f"{p.duration_ms:.4f}",
                    f"{r.speedup:.2f}",
                    f"{p.dram_bandwidth_gb_s:.2f}",
                    f"{r.bandwidth_ratio:.2f}",
                    f"{p.sm_throughput:.1f}",
                    f"{p.memory_throughput:.1f}",
                    f"{p.sectors_per_request:.2f}",
                    f"{p.l2_hit_rate:.1f}",
                    f"{p.achieved_occupancy:.1f}",
                    p.bottleneck.value,
                    "Yes" if r.regression else "No"
                ])

        print(f"✓ Comparison exported to: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Compare performance of multiple kernel versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 4 versions with v0 as baseline
  python compare_versions.py v0_naive v1_coalesced v2_vectorized v4_optimized

  # Specify different baseline
  python compare_versions.py v*.cu --baseline=v1_coalesced

  # Pass arguments to binaries
  python compare_versions.py gemm_v1 gemm_v2 -- 1024 1024 1024

  # Export results
  python compare_versions.py v*.cu --json=results.json --csv=results.csv
        """
    )

    parser.add_argument('binaries', nargs='+', help='Binary files to compare')
    parser.add_argument('--baseline', help='Baseline version (default: first binary)')
    parser.add_argument('--json', help='Export JSON to file')
    parser.add_argument('--csv', help='Export CSV to file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed comparison')

    args = parser.parse_args()

    # 检查文件存在
    binaries = []
    for binary in args.binaries:
        if os.path.exists(binary):
            binaries.append(binary)
        else:
            print(f"⚠️  Warning: {binary} not found, skipping")

    if not binaries:
        print("✗ Error: No valid binaries found")
        sys.exit(1)

    # 创建对比器
    comparator = VersionComparator()

    # 运行对比
    print(f"Comparing {len(binaries)} versions...")
    if args.baseline:
        print(f"Baseline: {args.baseline}")

    results = comparator.compare(binaries, baseline=args.baseline)

    # 打印结果
    comparator.print_comparison_table(args.baseline)

    if args.detailed:
        comparator.print_detailed_comparison()

    # 导出
    if args.json:
        comparator.export_json(args.json)

    if args.csv:
        comparator.export_csv(args.csv)


if __name__ == "__main__":
    main()
