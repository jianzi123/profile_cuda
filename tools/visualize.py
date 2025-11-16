#!/usr/bin/env python3
"""
Performance Visualization Tool

功能:
1. 生成性能对比柱状图
2. 生成优化历程趋势图
3. 生成 NCU 指标雷达图
4. 生成带宽利用率分析图

使用:
    python visualize.py <comparison.json>
    python visualize.py results.json --output=charts.png
"""

import json
import sys
import argparse
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class PerformanceVisualizer:
    """性能可视化工具"""

    def __init__(self, data_file: str):
        """
        Args:
            data_file: JSON 格式的对比数据文件
        """
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        """加载 JSON 数据"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"✗ Error: Data file not found: {self.data_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON: {e}")
            sys.exit(1)

    def plot_speedup_comparison(self, ax=None):
        """绘制加速比对比图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        versions = [item['name'] for item in self.data['comparison']]
        speedups = [item['speedup'] for item in self.data['comparison']]

        # 颜色: 基准=灰色, 加速>2x=绿色, 加速<2x=黄色, 回归=红色
        colors = []
        for i, item in enumerate(self.data['comparison']):
            if i == 0:
                colors.append('#808080')  # Gray for baseline
            elif item.get('regression', False):
                colors.append('#e74c3c')  # Red for regression
            elif speedups[i] > 2.0:
                colors.append('#27ae60')  # Green for good
            else:
                colors.append('#f39c12')  # Yellow for moderate

        bars = ax.bar(versions, speedups, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Speedup (vs Baseline)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Speedup Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(speedups) * 1.2)
        ax.grid(axis='y', alpha=0.3)

        # 旋转 x 轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        return ax

    def plot_bandwidth_comparison(self, ax=None):
        """绘制带宽对比图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        versions = [item['name'] for item in self.data['comparison']]
        bandwidths = [item['bandwidth_gb_s'] for item in self.data['comparison']]

        # 理论峰值 (A100: 1555 GB/s)
        theoretical_bw = 1555  # TODO: 从数据中读取

        bars = ax.bar(versions, bandwidths, color='#3498db', alpha=0.8, edgecolor='black')

        # 添加理论峰值线
        ax.axhline(y=theoretical_bw, color='red', linestyle='--',
                  linewidth=2, label=f'Theoretical Peak ({theoretical_bw} GB/s)')

        # 添加数值标签和效率
        for bar, bw in zip(bars, bandwidths):
            height = bar.get_height()
            efficiency = (bw / theoretical_bw) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{bw:.0f}\n({efficiency:.1f}%)',
                   ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Bandwidth Utilization', fontsize=14, fontweight='bold')
        ax.set_ylim(0, theoretical_bw * 1.15)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        return ax

    def plot_optimization_timeline(self, ax=None):
        """绘制优化时间线 (时间递减图)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        versions = [item['name'] for item in self.data['comparison']]
        times = [item['time_ms'] for item in self.data['comparison']]

        # 绘制线图
        ax.plot(versions, times, marker='o', linewidth=2, markersize=8,
               color='#e74c3c', label='Execution Time')

        # 填充区域
        ax.fill_between(range(len(versions)), times, alpha=0.3, color='#e74c3c')

        # 添加数值标签
        for i, (v, t) in enumerate(zip(versions, times)):
            ax.text(i, t, f'{t:.2f}ms', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Optimization Version', fontsize=12, fontweight='bold')
        ax.set_title('Optimization Timeline (Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        return ax

    def plot_metrics_radar(self, version_indices: List[int] = None):
        """
        绘制 NCU 指标雷达图

        Args:
            version_indices: 要对比的版本索引列表 (默认全部)
        """
        if version_indices is None:
            version_indices = list(range(len(self.data['comparison'])))

        # 指标列表
        metrics = ['SM %', 'Memory %', 'Occupancy %', 'BW Efficiency %']

        # 设置雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']

        for idx, ver_idx in enumerate(version_indices):
            item = self.data['comparison'][ver_idx]

            # 提取指标值
            values = [
                item['sm_throughput'],
                item['memory_throughput'],
                item['occupancy'],
                (item['bandwidth_gb_s'] / 1555) * 100  # 假设 A100
            ]
            values += values[:1]  # 闭合

            # 绘制
            ax.plot(angles, values, 'o-', linewidth=2, label=item['name'],
                   color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('NCU Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)

        return ax

    def plot_bottleneck_distribution(self, ax=None):
        """绘制瓶颈类型分布饼图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # 统计瓶颈类型
        bottlenecks = [item['bottleneck'] for item in self.data['comparison']]
        bottleneck_counts = {}
        for b in bottlenecks:
            bottleneck_counts[b] = bottleneck_counts.get(b, 0) + 1

        labels = list(bottleneck_counts.keys())
        sizes = list(bottleneck_counts.values())

        colors_map = {
            'Memory-bound': '#e74c3c',
            'Compute-bound': '#3498db',
            'Launch-bound': '#f39c12',
            'Balanced': '#27ae60',
            'Unknown': '#95a5a6'
        }
        colors = [colors_map.get(label, '#95a5a6') for label in labels]

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
              startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Bottleneck Type Distribution', fontsize=14, fontweight='bold')

        return ax

    def generate_all_charts(self, output_file: str = 'performance_charts.png'):
        """生成包含所有图表的综合视图"""
        fig = plt.figure(figsize=(20, 12))

        # 2x3 布局
        ax1 = plt.subplot(2, 3, 1)
        self.plot_speedup_comparison(ax1)

        ax2 = plt.subplot(2, 3, 2)
        self.plot_bandwidth_comparison(ax2)

        ax3 = plt.subplot(2, 3, 3)
        self.plot_optimization_timeline(ax3)

        ax4 = plt.subplot(2, 3, 4)
        self.plot_bottleneck_distribution(ax4)

        # 雷达图 - 对比第一个和最后一个版本
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        n_versions = len(self.data['comparison'])
        indices_to_plot = [0, n_versions-1] if n_versions > 1 else [0]

        metrics = ['SM %', 'Memory %', 'Occupancy %', 'BW Eff %']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        colors = ['#e74c3c', '#27ae60']
        for idx, ver_idx in enumerate(indices_to_plot):
            item = self.data['comparison'][ver_idx]
            values = [
                item['sm_throughput'],
                item['memory_throughput'],
                item['occupancy'],
                (item['bandwidth_gb_s'] / 1555) * 100
            ]
            values += values[:1]

            ax5.plot(angles, values, 'o-', linewidth=2, label=item['name'],
                    color=colors[idx])
            ax5.fill(angles, values, alpha=0.15, color=colors[idx])

        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics)
        ax5.set_ylim(0, 100)
        ax5.set_yticks([20, 40, 60, 80, 100])
        ax5.legend(loc='upper right')
        ax5.set_title('Baseline vs Optimized', fontweight='bold')
        ax5.grid(True)

        # 指标对比表格
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # 创建表格数据
        table_data = [['Version', 'Time (ms)', 'Speedup', 'BW (GB/s)', 'Bottleneck']]
        for item in self.data['comparison']:
            table_data.append([
                item['name'][:15],  # 截断长名称
                f"{item['time_ms']:.2f}",
                f"{item['speedup']:.2f}x",
                f"{item['bandwidth_gb_s']:.0f}",
                item['bottleneck'][:12]
            ])

        table = ax6.table(cellText=table_data, cellLoc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # 表头加粗
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Charts saved to: {output_file}")

        return fig


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Visualize kernel performance comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('data_file', help='JSON data file from compare_versions.py')
    parser.add_argument('--output', '-o', default='performance_charts.png',
                       help='Output image file (default: performance_charts.png)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='Output format')

    args = parser.parse_args()

    # 调整输出文件扩展名
    output_file = Path(args.output).with_suffix(f'.{args.format}')

    print(f"Loading data from: {args.data_file}")
    visualizer = PerformanceVisualizer(args.data_file)

    print(f"Generating charts...")
    visualizer.generate_all_charts(str(output_file))

    print(f"\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
