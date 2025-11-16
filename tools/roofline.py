#!/usr/bin/env python3
"""
Automated Roofline Model Analysis Tool

功能:
1. 自动计算 Arithmetic Intensity (AI)
2. 生成 Roofline 图表
3. 判断 Memory-bound 还是 Compute-bound
4. 提供优化建议

使用:
    python roofline.py --flops=<flops> --bytes=<bytes> --time=<time_ms>
    python roofline.py --kernel=<binary>  # 自动从 NCU 提取
    python roofline.py --help
"""

import argparse
import subprocess
import json
import sys
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class GPUModel(Enum):
    """GPU 型号及其硬件参数"""
    A100 = {
        'name': 'NVIDIA A100',
        'peak_fp32_tflops': 19.5,
        'peak_fp16_tflops': 312,  # With Tensor Core
        'peak_bandwidth_gb_s': 1555,
        'ridge_point_fp32': 19.5 * 1000 / 1555,  # ~12.5 FLOPS/Byte
        'ridge_point_fp16': 312 * 1000 / 1555,   # ~200 FLOPS/Byte
    }
    V100 = {
        'name': 'NVIDIA V100',
        'peak_fp32_tflops': 15.7,
        'peak_fp16_tflops': 125,
        'peak_bandwidth_gb_s': 900,
        'ridge_point_fp32': 15.7 * 1000 / 900,
        'ridge_point_fp16': 125 * 1000 / 900,
    }
    RTX3090 = {
        'name': 'NVIDIA RTX 3090',
        'peak_fp32_tflops': 35.6,
        'peak_fp16_tflops': 71,
        'peak_bandwidth_gb_s': 936,
        'ridge_point_fp32': 35.6 * 1000 / 936,
        'ridge_point_fp16': 71 * 1000 / 936,
    }


@dataclass
class RooflineResult:
    """Roofline 分析结果"""
    arithmetic_intensity: float  # FLOPS/Byte
    achieved_gflops: float
    achieved_bandwidth_gb_s: float
    gpu_model: str
    peak_compute_gflops: float
    peak_bandwidth_gb_s: float
    ridge_point: float
    bottleneck: str  # "Memory-bound" or "Compute-bound"
    ceiling: float  # 理论性能上界 (GFLOPS)
    efficiency: float  # 实际性能 / 理论上界
    optimization_suggestions: list

    def __str__(self):
        return f"""
{'=' * 70}
Roofline Model Analysis
{'=' * 70}

GPU: {self.gpu_model}
  Peak Compute: {self.peak_compute_gflops:.1f} GFLOPS
  Peak Bandwidth: {self.peak_bandwidth_gb_s:.1f} GB/s
  Ridge Point: {self.ridge_point:.2f} FLOPS/Byte

Kernel Performance:
  Arithmetic Intensity (AI): {self.arithmetic_intensity:.3f} FLOPS/Byte
  Achieved Performance: {self.achieved_gflops:.2f} GFLOPS
  Achieved Bandwidth: {self.achieved_bandwidth_gb_s:.2f} GB/s

Analysis:
  Bottleneck: {self.bottleneck}
  Performance Ceiling: {self.ceiling:.2f} GFLOPS
  Efficiency: {self.efficiency:.1f}%

Optimization Suggestions:
""" + '\n'.join(f"  • {s}" for s in self.optimization_suggestions)


class RooflineAnalyzer:
    """Roofline 模型分析器"""

    def __init__(self, gpu_model: GPUModel = GPUModel.A100, precision: str = 'fp32'):
        """
        Args:
            gpu_model: GPU 型号
            precision: 精度 ('fp32' or 'fp16')
        """
        self.gpu = gpu_model.value
        self.precision = precision

        # 选择对应精度的峰值性能
        if precision == 'fp32':
            self.peak_compute = self.gpu['peak_fp32_tflops'] * 1000  # GFLOPS
            self.ridge_point = self.gpu['ridge_point_fp32']
        else:  # fp16
            self.peak_compute = self.gpu['peak_fp16_tflops'] * 1000
            self.ridge_point = self.gpu['ridge_point_fp16']

        self.peak_bandwidth = self.gpu['peak_bandwidth_gb_s']

    def analyze(
        self,
        flops: float,
        bytes_accessed: float,
        time_ms: float
    ) -> RooflineResult:
        """
        Roofline 分析

        Args:
            flops: 总浮点操作数
            bytes_accessed: 总内存访问字节数
            time_ms: 执行时间 (毫秒)

        Returns:
            RooflineResult: 分析结果
        """
        # 计算 Arithmetic Intensity
        ai = flops / bytes_accessed if bytes_accessed > 0 else 0

        # 计算实际性能
        time_s = time_ms / 1000.0
        achieved_gflops = (flops / 1e9) / time_s if time_s > 0 else 0
        achieved_bandwidth = (bytes_accessed / 1e9) / time_s if time_s > 0 else 0

        # 判断瓶颈
        if ai < self.ridge_point:
            bottleneck = "Memory-bound"
            # 理论上界 = AI × Peak Bandwidth
            ceiling = ai * self.peak_bandwidth
        else:
            bottleneck = "Compute-bound"
            # 理论上界 = Peak Compute
            ceiling = self.peak_compute

        # 计算效率
        efficiency = (achieved_gflops / ceiling * 100) if ceiling > 0 else 0

        # 生成优化建议
        suggestions = self._generate_suggestions(
            ai, bottleneck, efficiency, achieved_bandwidth
        )

        return RooflineResult(
            arithmetic_intensity=ai,
            achieved_gflops=achieved_gflops,
            achieved_bandwidth_gb_s=achieved_bandwidth,
            gpu_model=self.gpu['name'],
            peak_compute_gflops=self.peak_compute,
            peak_bandwidth_gb_s=self.peak_bandwidth,
            ridge_point=self.ridge_point,
            bottleneck=bottleneck,
            ceiling=ceiling,
            efficiency=efficiency,
            optimization_suggestions=suggestions
        )

    def _generate_suggestions(
        self,
        ai: float,
        bottleneck: str,
        efficiency: float,
        achieved_bw: float
    ) -> list:
        """生成优化建议"""
        suggestions = []

        if bottleneck == "Memory-bound":
            suggestions.append(
                f"🔴 Kernel 是 Memory-bound (AI={ai:.3f} < Ridge Point={self.ridge_point:.2f})"
            )

            if efficiency < 50:
                suggestions.append(
                    "内存访问效率低，优化方向:"
                    "\n       1. 检查 memory coalescing (NCU: sectors_per_request)"
                    "\n       2. 使用 float4 向量化访问"
                    "\n       3. 考虑 Shared Memory (如果有数据重用)"
                )
            elif efficiency < 80:
                suggestions.append(
                    "内存访问已优化，但仍受带宽限制:"
                    "\n       1. Kernel Fusion (减少显存往返)"
                    "\n       2. 增加 Arithmetic Intensity (更多计算/数据)"
                    "\n       3. 考虑混合精度 (FP16 减少带宽需求)"
                )
            else:
                suggestions.append(
                    f"✅ 内存带宽已接近极限 (效率 {efficiency:.1f}%)"
                    "\n       → 无需进一步优化内核"
                    "\n       → 考虑算子融合或系统级优化"
                )

            # 估算提升空间
            potential_speedup = self.peak_bandwidth / achieved_bw
            suggestions.append(
                f"理论提升空间: {potential_speedup:.2f}x (如果达到 100% 带宽)"
            )

        else:  # Compute-bound
            suggestions.append(
                f"🔵 Kernel 是 Compute-bound (AI={ai:.3f} > Ridge Point={self.ridge_point:.2f})"
            )

            if efficiency < 50:
                suggestions.append(
                    "计算效率低，优化方向:"
                    "\n       1. 使用 Tensor Core (FP16/TF32 for GEMM)"
                    "\n       2. 增加 ILP (指令级并行)"
                    "\n       3. Loop Unrolling"
                    "\n       4. 检查 warp divergence"
                )
            elif efficiency < 80:
                suggestions.append(
                    "计算已优化，进一步提升:"
                    "\n       1. Tensor Core (如果是矩阵乘法)"
                    "\n       2. 混合精度 (FP16 性能 ~16x FP32)"
                    "\n       3. 算子融合"
                )
            else:
                suggestions.append(
                    f"✅ 计算效率已接近极限 (效率 {efficiency:.1f}%)"
                    "\n       → 考虑 Tensor Core 或混合精度"
                )

        # 通用建议
        if ai < 0.5:
            suggestions.append(
                f"⚠️  AI 非常低 ({ai:.3f}), 考虑:"
                "\n       • 算子融合 (减少内存访问)"
                "\n       • 增加计算量 (如果业务允许)"
            )

        return suggestions

    def plot_roofline(
        self,
        results: list,
        output_file: str = 'roofline.png',
        show_points: bool = True
    ):
        """
        绘制 Roofline 图

        Args:
            results: RooflineResult 列表
            output_file: 输出文件名
            show_points: 是否显示实际性能点
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # AI 范围
        ai_range = np.logspace(-2, 3, 1000)  # 0.01 to 1000 FLOPS/Byte

        # Memory-bound 区域 (斜线): Performance = AI × Peak BW
        memory_bound = ai_range * self.peak_bandwidth
        memory_bound = np.minimum(memory_bound, self.peak_compute)

        # Compute-bound 区域 (平线): Performance = Peak Compute
        compute_bound = np.ones_like(ai_range) * self.peak_compute

        # Roofline (取最小值)
        roofline = np.minimum(memory_bound, compute_bound)

        # 绘制 Roofline
        ax.loglog(ai_range, roofline, 'k-', linewidth=3, label='Roofline')

        # 填充区域
        ax.fill_between(ai_range, roofline, alpha=0.2, color='gray')

        # Ridge Point 标记
        ax.axvline(x=self.ridge_point, color='red', linestyle='--',
                  linewidth=2, alpha=0.7,
                  label=f'Ridge Point ({self.ridge_point:.1f} FLOPS/Byte)')

        # 绘制实际性能点
        if show_points and results:
            colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']

            for i, result in enumerate(results):
                color = colors[i % len(colors)]
                label = getattr(result, 'label', f'Kernel {i+1}')

                ax.scatter(result.arithmetic_intensity,
                          result.achieved_gflops,
                          s=200, color=color, marker='o',
                          edgecolors='black', linewidth=2,
                          label=label, zorder=5)

                # 添加标签
                ax.annotate(
                    f'{label}\n({result.efficiency:.0f}%)',
                    xy=(result.arithmetic_intensity, result.achieved_gflops),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        # 设置坐标轴
        ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Performance (GFLOPS)',
                     fontsize=14, fontweight='bold')
        ax.set_title(f'Roofline Model - {self.gpu["name"]} ({self.precision.upper()})',
                    fontsize=16, fontweight='bold')

        ax.set_xlim([0.01, 1000])
        ax.set_ylim([1, self.peak_compute * 2])

        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)

        # 添加区域标注
        ax.text(0.02, self.peak_bandwidth * 0.02,
               'Memory-Bound\nRegion',
               fontsize=12, style='italic', alpha=0.7,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        ax.text(self.ridge_point * 10, self.peak_compute * 0.8,
               'Compute-Bound\nRegion',
               fontsize=12, style='italic', alpha=0.7,
               bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3))

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Roofline chart saved to: {output_file}")

        return fig


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Roofline Model Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Manual input
  python roofline.py --flops=1e9 --bytes=256e6 --time=1.5

  # From NCU profile
  python roofline.py --kernel=./vector_add

  # Multiple kernels comparison
  python roofline.py --kernels v0_naive v1_coalesced v4_optimized --plot

  # Specify GPU model
  python roofline.py --flops=1e9 --bytes=256e6 --time=1.5 --gpu=V100 --precision=fp16
        """
    )

    parser.add_argument('--flops', type=float, help='Total floating point operations')
    parser.add_argument('--bytes', type=float, help='Total bytes accessed')
    parser.add_argument('--time', type=float, help='Execution time in ms')

    parser.add_argument('--kernel', help='Kernel binary (auto-extract from NCU)')
    parser.add_argument('--kernels', nargs='+', help='Multiple kernels for comparison')

    parser.add_argument('--gpu', choices=['A100', 'V100', 'RTX3090'],
                       default='A100', help='GPU model')
    parser.add_argument('--precision', choices=['fp32', 'fp16'],
                       default='fp32', help='Precision')

    parser.add_argument('--plot', action='store_true', help='Generate Roofline plot')
    parser.add_argument('--output', default='roofline.png', help='Output plot file')

    args = parser.parse_args()

    # 选择 GPU 型号
    gpu_enum = getattr(GPUModel, args.gpu)
    analyzer = RooflineAnalyzer(gpu_enum, args.precision)

    results = []

    # 模式 1: 手动输入
    if args.flops and args.bytes and args.time:
        result = analyzer.analyze(args.flops, args.bytes, args.time)
        result.label = 'User Kernel'
        results.append(result)
        print(result)

    # 模式 2: 单个 kernel (TODO: 从 NCU 提取)
    elif args.kernel:
        print("⚠️  Auto-extraction from NCU not yet implemented")
        print("   Please use --flops, --bytes, --time for now")
        sys.exit(1)

    # 模式 3: 多个 kernels 对比 (TODO)
    elif args.kernels:
        print("⚠️  Multi-kernel comparison not yet implemented")
        print("   Please use --flops, --bytes, --time for now")
        sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

    # 绘图
    if args.plot and results:
        analyzer.plot_roofline(results, args.output)


if __name__ == "__main__":
    main()
