#!/usr/bin/env python3
"""
NCU 报告自动分析脚本

使用方法：
1. 运行 NCU 并导出 CSV：
   ncu --csv --page raw ./program > ncu_output.csv

2. 运行此脚本分析：
   python analyze_ncu.py ncu_output.csv
"""

import sys
import csv
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class BottleneckType(Enum):
    COMPUTE_BOUND = "Compute-bound"
    MEMORY_BOUND = "Memory-bound"
    BALANCED = "Balanced"
    UNDER_UTILIZED = "Under-utilized"


@dataclass
class Metric:
    name: str
    value: float
    unit: str


class NCUAnalyzer:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.metrics: Dict[str, Metric] = {}
        self.kernel_name = ""
        self.duration_ns = 0

    def parse_csv(self):
        """解析 NCU CSV 输出"""
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric_name = row.get('Metric Name', '')
                metric_value = row.get('Metric Value', '')
                metric_unit = row.get('Metric Unit', '')

                # 转换为数值
                try:
                    value = float(metric_value.replace(',', ''))
                except (ValueError, AttributeError):
                    value = 0.0

                self.metrics[metric_name] = Metric(metric_name, value, metric_unit)

                # 提取 kernel 名称
                if 'Kernel Name' in row:
                    self.kernel_name = row['Kernel Name']

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """获取指标值"""
        return self.metrics.get(name, Metric(name, default, "")).value

    def analyze_bottleneck(self) -> BottleneckType:
        """第一步：判断瓶颈类型"""
        sm_throughput = self.get_metric('sm__throughput.avg.pct_of_peak_sustained_elapsed')
        mem_throughput = self.get_metric('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed')

        print("=" * 80)
        print("第一步：Speed of Light 分析")
        print("=" * 80)
        print(f"SM Throughput:          {sm_throughput:6.2f}%")
        print(f"Memory Throughput:      {mem_throughput:6.2f}%")
        print()

        # 判断瓶颈类型
        if sm_throughput > 80 and mem_throughput < 60:
            bottleneck = BottleneckType.COMPUTE_BOUND
            print(f"结论: {bottleneck.value} ✓")
            print("计算单元高度利用，内存不是瓶颈")
        elif sm_throughput < 60 and mem_throughput > 80:
            bottleneck = BottleneckType.MEMORY_BOUND
            print(f"结论: {bottleneck.value} ✓")
            print("内存带宽接近饱和，计算单元空闲")
        elif sm_throughput > 70 and mem_throughput > 70:
            bottleneck = BottleneckType.BALANCED
            print(f"结论: {bottleneck.value} ✓")
            print("计算和内存都接近极限，优化困难")
        else:
            bottleneck = BottleneckType.UNDER_UTILIZED
            print(f"结论: {bottleneck.value} ⚠")
            print("计算和内存都未充分利用，需要进一步分析")

        print()
        return bottleneck

    def analyze_memory(self):
        """第二步：Memory-bound 深入分析"""
        print("=" * 80)
        print("第二步：Memory 深入分析")
        print("=" * 80)

        # DRAM 带宽
        dram_throughput_pct = self.get_metric('dram__throughput.avg.pct_of_peak_sustained_elapsed')
        dram_bytes_read = self.get_metric('dram__bytes_read.sum')
        dram_bytes_write = self.get_metric('dram__bytes_write.sum')
        duration_ns = self.get_metric('gpu__time_duration.sum')

        if duration_ns > 0:
            total_bytes = dram_bytes_read + dram_bytes_write
            duration_sec = duration_ns / 1e9
            actual_bw_gbps = total_bytes / duration_sec / 1e9

            # 假设 A100，理论带宽 1555 GB/s
            theoretical_bw = 1555
            bw_util = (actual_bw_gbps / theoretical_bw) * 100

            print(f"\n📊 DRAM 带宽分析:")
            print(f"  DRAM Throughput:      {dram_throughput_pct:6.2f}%")
            print(f"  读取字节数:           {dram_bytes_read / 1e9:8.2f} GB")
            print(f"  写入字节数:           {dram_bytes_write / 1e9:8.2f} GB")
            print(f"  实际带宽:             {actual_bw_gbps:8.2f} GB/s")
            print(f"  理论带宽:             {theoretical_bw:8.2f} GB/s (A100)")
            print(f"  带宽利用率:           {bw_util:6.2f}%")

            if bw_util < 80:
                print(f"  ⚠️  带宽利用率低于 80%，有优化空间")
            else:
                print(f"  ✓ 带宽利用率高，接近硬件极限")

        # L2 Cache
        l2_hit_rate = self.get_metric('lts__t_sector_hit_rate.pct')
        print(f"\n📊 L2 Cache 分析:")
        print(f"  L2 命中率:            {l2_hit_rate:6.2f}%")

        if l2_hit_rate > 70:
            print(f"  ✓ L2 命中率良好，数据有复用")
        elif l2_hit_rate > 50:
            print(f"  ⚠️  L2 命中率一般，有改进空间")
        else:
            print(f"  ❌ L2 命中率低，数据几乎没有复用")
            print(f"     建议：算子融合、Tiling、增加数据复用")

        # L1 Cache
        l1_hit_rate = self.get_metric('l1tex__t_sector_hit_rate.pct')
        print(f"\n📊 L1/TEX Cache 分析:")
        print(f"  L1 命中率:            {l1_hit_rate:6.2f}%")

        if l1_hit_rate > 90:
            print(f"  ✓ L1 命中率优秀")
        elif l1_hit_rate > 70:
            print(f"  ⚠️  L1 命中率一般")
        else:
            print(f"  ❌ L1 命中率低")
            print(f"     建议：使用 Shared Memory、提高数据局部性")

        # Coalesced Access
        coalesced_ld = self.get_metric(
            'l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio'
        )
        coalesced_st = self.get_metric(
            'l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio'
        )

        print(f"\n📊 Coalesced Access 分析:")
        print(f"  加载 Coalesced 程度:  {coalesced_ld:6.2f} (理想值 = 1.0)")
        print(f"  存储 Coalesced 程度:  {coalesced_st:6.2f} (理想值 = 1.0)")

        if coalesced_ld <= 1.2 and coalesced_st <= 1.2:
            print(f"  ✓ 内存访问模式良好，接近完美合并")
        elif coalesced_ld <= 2.0 or coalesced_st <= 2.0:
            waste = ((max(coalesced_ld, coalesced_st) - 1.0) / max(coalesced_ld, coalesced_st)) * 100
            print(f"  ⚠️  有非合并访问，浪费约 {waste:.1f}% 带宽")
            print(f"     建议：调整访问模式、使用向量化 (float4)")
        else:
            waste = ((max(coalesced_ld, coalesced_st) - 1.0) / max(coalesced_ld, coalesced_st)) * 100
            print(f"  ❌ 严重的非合并访问，浪费约 {waste:.1f}% 带宽")
            print(f"     建议：重新设计内存访问模式、使用 Shared Memory")

        # Bank Conflicts
        bank_conflicts = self.get_metric('l1tex__data_bank_conflicts_pipe_lsu.sum')
        shared_accesses = self.get_metric('l1tex__data_pipe_lsu_wavefronts_mem_shared.sum')

        if shared_accesses > 0:
            conflict_rate = (bank_conflicts / shared_accesses) * 100
            print(f"\n📊 Shared Memory Bank Conflicts:")
            print(f"  Bank Conflicts:       {bank_conflicts:10.0f}")
            print(f"  Shared Accesses:      {shared_accesses:10.0f}")
            print(f"  Conflict 率:          {conflict_rate:6.2f}%")

            if conflict_rate < 1:
                print(f"  ✓ 几乎无 bank conflicts")
            elif conflict_rate < 10:
                print(f"  ⚠️  有少量 bank conflicts")
                print(f"     建议：考虑添加 padding 到 shared memory 数组")
            else:
                print(f"  ❌ 严重的 bank conflicts")
                print(f"     建议：重新组织 shared memory 访问模式、添加 padding")

        print()

    def analyze_compute(self):
        """第二步：Compute-bound 深入分析"""
        print("=" * 80)
        print("第二步：Compute 深入分析")
        print("=" * 80)

        # Warp Divergence
        threads_per_inst = self.get_metric('smsp__thread_inst_executed_per_inst_executed.ratio')
        print(f"\n📊 Warp Divergence 分析:")
        print(f"  平均每指令执行线程数: {threads_per_inst:6.2f} (理想值 = 32)")

        if threads_per_inst >= 30:
            print(f"  ✓ 几乎无 warp divergence")
        elif threads_per_inst >= 24:
            waste = ((32 - threads_per_inst) / 32) * 100
            print(f"  ⚠️  有轻微 divergence，浪费约 {waste:.1f}% 计算")
            print(f"     建议：检查分支代码，尽量让 warp 内线程执行相同路径")
        else:
            waste = ((32 - threads_per_inst) / 32) * 100
            print(f"  ❌ 严重的 warp divergence，浪费约 {waste:.1f}% 计算")
            print(f"     建议：重组数据、减少条件分支、使用 warp 原语")

        # ILP
        issue_active = self.get_metric('smsp__issue_active.avg.pct_of_peak_sustained_active')
        print(f"\n📊 ILP (指令级并行) 分析:")
        print(f"  指令发射活跃度:       {issue_active:6.2f}%")

        if issue_active > 70:
            print(f"  ✓ ILP 利用率良好")
        elif issue_active > 50:
            print(f"  ⚠️  ILP 有改进空间")
            print(f"     建议：每个线程处理多个数据、循环展开")
        else:
            print(f"  ❌ ILP 利用率低")
            print(f"     建议：向量化、手动展开循环、增加独立操作")

        # FLOPs 统计
        fadd = self.get_metric('smsp__sass_thread_inst_executed_op_fadd_pred_on.sum')
        fmul = self.get_metric('smsp__sass_thread_inst_executed_op_fmul_pred_on.sum')
        ffma = self.get_metric('smsp__sass_thread_inst_executed_op_ffma_pred_on.sum')
        duration_ns = self.get_metric('gpu__time_duration.sum')

        if duration_ns > 0 and (fadd + fmul + ffma) > 0:
            fp32_flops = fadd + fmul + 2 * ffma
            duration_sec = duration_ns / 1e9
            tflops = fp32_flops / duration_sec / 1e12

            # A100 理论峰值：19.5 TFLOPS
            peak_tflops = 19.5
            efficiency = (tflops / peak_tflops) * 100

            print(f"\n📊 FP32 性能分析:")
            print(f"  FP32 FLOPs:           {fp32_flops / 1e9:10.2f} GFLOPs")
            print(f"  实际吞吐量:           {tflops:10.3f} TFLOPS")
            print(f"  理论峰值:             {peak_tflops:10.3f} TFLOPS (A100)")
            print(f"  计算效率:             {efficiency:6.2f}%")

        # Tensor Core 检查
        tensor_active = self.get_metric('smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active')
        if tensor_active > 0:
            print(f"\n📊 Tensor Core 分析:")
            print(f"  Tensor Core 利用率:   {tensor_active:6.2f}%")
            print(f"  ✓ 正在使用 Tensor Cores")
        else:
            # 检查是否有 FP16 操作
            hadd = self.get_metric('smsp__sass_thread_inst_executed_op_hadd_pred_on.sum')
            hmul = self.get_metric('smsp__sass_thread_inst_executed_op_hmul_pred_on.sum')
            hfma = self.get_metric('smsp__sass_thread_inst_executed_op_hfma_pred_on.sum')

            if (hadd + hmul + hfma) > fp32_flops * 0.5:
                print(f"\n📊 Tensor Core 机会:")
                print(f"  ⚠️  检测到大量 FP16 计算，但未使用 Tensor Cores")
                print(f"     建议：考虑使用 WMMA API 或 cuBLAS")

        print()

    def analyze_occupancy(self):
        """第二步：Occupancy 分析"""
        print("=" * 80)
        print("第二步：Occupancy 分析")
        print("=" * 80)

        achieved_occ = self.get_metric('sm__warps_active.avg.pct_of_peak_sustained_active')

        print(f"\n📊 占用率分析:")
        print(f"  实际占用率:           {achieved_occ:6.2f}%")

        if achieved_occ > 60:
            print(f"  ✓ 占用率良好")
        elif achieved_occ > 40:
            print(f"  ⚠️  占用率中等，有提升空间")
        else:
            print(f"  ❌ 占用率低，严重限制性能")
            print(f"     建议：检查资源使用、调整 block size")

        print()

    def analyze_warp_state(self):
        """Warp 停顿分析"""
        print("=" * 80)
        print("Warp State 分析")
        print("=" * 80)

        stall_barrier = self.get_metric('smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active')
        stall_long = self.get_metric('smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active')
        stall_short = self.get_metric('smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active')
        stall_not_selected = self.get_metric('smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active')

        print(f"\n📊 Warp 停顿分布:")
        print(f"  Barrier (__syncthreads):  {stall_barrier:6.2f}%")
        print(f"  Long Scoreboard (Memory): {stall_long:6.2f}%")
        print(f"  Short Scoreboard (Compute):{stall_short:6.2f}%")
        print(f"  Not Selected (Occupancy): {stall_not_selected:6.2f}%")

        # 找出主要停顿原因
        stalls = {
            'Barrier': stall_barrier,
            'Memory Latency': stall_long,
            'Compute Dependency': stall_short,
            'Low Occupancy': stall_not_selected,
        }

        max_stall_name = max(stalls, key=stalls.get)
        max_stall_value = stalls[max_stall_name]

        print(f"\n  主要停顿原因: {max_stall_name} ({max_stall_value:.2f}%)")

        if max_stall_name == 'Barrier' and max_stall_value > 30:
            print(f"  建议：减少 __syncthreads() 调用频率")
        elif max_stall_name == 'Memory Latency' and max_stall_value > 50:
            print(f"  建议：优化内存访问、提高缓存命中率")
        elif max_stall_name == 'Compute Dependency' and max_stall_value > 30:
            print(f"  建议：提高 ILP、减少数据依赖")
        elif max_stall_name == 'Low Occupancy' and max_stall_value > 40:
            print(f"  建议：增加 block 数量、优化资源使用")

        print()

    def generate_optimization_suggestions(self, bottleneck: BottleneckType):
        """生成优化建议"""
        print("=" * 80)
        print("💡 优化建议总结")
        print("=" * 80)

        if bottleneck == BottleneckType.MEMORY_BOUND:
            print("\n✅ 优先级 1（最重要）：")
            print("  1. 算子融合：将多个 kernel 合并，减少内存访问")
            print("  2. 使用 Shared Memory：缓存频繁访问的数据")

            print("\n✅ 优先级 2：")
            print("  3. 向量化访问：使用 float4 提高带宽利用率")
            print("  4. 优化访问模式：确保 coalesced access")

            print("\n✅ 优先级 3：")
            print("  5. 提高缓存命中率：Tiling、增加数据复用")
            print("  6. 消除 Bank Conflicts：添加 padding")

        elif bottleneck == BottleneckType.COMPUTE_BOUND:
            print("\n✅ 优先级 1（最重要）：")
            print("  1. 使用 Tensor Cores：如果是矩阵运算")
            print("  2. 减少 Warp Divergence：重组数据、减少分支")

            print("\n✅ 优先级 2：")
            print("  3. 提高 ILP：每个线程处理多个数据")
            print("  4. 使用快速数学函数：__expf, __logf 等")

            print("\n✅ 优先级 3：")
            print("  5. 循环展开：#pragma unroll")
            print("  6. 算法优化：减少不必要的计算")

        elif bottleneck == BottleneckType.UNDER_UTILIZED:
            print("\n✅ 优先级 1（最重要）：")
            print("  1. 提高 Occupancy：调整 block size、减少资源使用")
            print("  2. 检查 Launch Configuration：确保有足够的 blocks")

            print("\n✅ 优先级 2：")
            print("  3. 分析 Warp State：找出主要停顿原因")
            print("  4. 减少同步开销：减少 __syncthreads()")

        print()

    def run_analysis(self):
        """运行完整分析"""
        print(f"\n{'=' * 80}")
        print(f"NCU 性能分析报告")
        print(f"{'=' * 80}")
        if self.kernel_name:
            print(f"Kernel: {self.kernel_name}")
        print()

        # 解析 CSV
        self.parse_csv()

        # 第一步：判断瓶颈
        bottleneck = self.analyze_bottleneck()

        # 第二步：深入分析
        if bottleneck == BottleneckType.MEMORY_BOUND:
            self.analyze_memory()
        elif bottleneck == BottleneckType.COMPUTE_BOUND:
            self.analyze_compute()
        elif bottleneck == BottleneckType.UNDER_UTILIZED:
            self.analyze_occupancy()
            self.analyze_warp_state()
        else:  # BALANCED
            print("Kernel 性能已接近硬件极限，优化空间有限")
            print("可以考虑从算法层面优化")
            print()

        # 生成优化建议
        self.generate_optimization_suggestions(bottleneck)


def main():
    if len(sys.argv) < 2:
        print("使用方法：")
        print("  1. 运行 NCU 并导出 CSV：")
        print("     ncu --csv --page raw ./program > ncu_output.csv")
        print("")
        print("  2. 分析 CSV：")
        print("     python analyze_ncu.py ncu_output.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    analyzer = NCUAnalyzer(csv_file)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
