# NVIDIA Nsight Compute (NCU) 完整使用指南

## 目录
1. [NCU 简介](#ncu-简介)
2. [安装与配置](#安装与配置)
3. [基础使用](#基础使用)
4. [核心指标解析](#核心指标解析)
5. [高级分析技巧](#高级分析技巧)
6. [实战案例](#实战案例)
7. [性能优化工作流](#性能优化工作流)

---

## NCU 简介

### 什么是 Nsight Compute？

NVIDIA Nsight Compute (NCU) 是一个交互式的 CUDA kernel 性能分析工具，提供详细的性能指标和优化建议。

**主要功能：**
- Kernel 级别的详细性能分析
- 内存吞吐量和带宽分析
- 计算单元利用率
- Warp 调度分析
- Roofline 分析
- 源代码级别的性能热点定位

**vs Nsight Systems：**
- **Nsight Systems**：系统级别，时间线分析，全局视角
- **Nsight Compute**：Kernel 级别，详细指标，微观优化

---

## 安装与配置

### 1. 安装

```bash
# Ubuntu/Debian
# NCU 通常随 CUDA Toolkit 一起安装
which ncu
# /usr/local/cuda/bin/ncu

# 单独下载
# https://developer.nvidia.com/nsight-compute

# 检查版本
ncu --version
```

### 2. 环境配置

```bash
# 添加到 PATH
export PATH=/usr/local/cuda/bin:$PATH

# 对于远程服务器，可能需要禁用 GUI
export DISPLAY=

# 提高权限（某些指标需要管理员权限）
# 临时方式
sudo ncu ...

# 永久方式（不推荐生产环境）
sudo su
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
update-initramfs -u
reboot
```

---

## 基础使用

### 1. 快速开始

```bash
# 基础 profile
ncu ./my_program

# Profile 特定 kernel
ncu --kernel-name matmul ./my_program

# Profile 前 N 次 kernel 调用
ncu --kernel-id ::matmul:1 ./my_program  # 第一次调用
ncu --kernel-id ::matmul:1:2 ./my_program  # 第一到第二次

# 保存结果
ncu -o profile_result ./my_program

# 加载结果（GUI）
ncu-ui profile_result.ncu-rep
```

### 2. 常用选项

```bash
# 完整指标集
ncu --set full -o full_profile ./my_program

# 指定指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./my_program

# 多个 section
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./my_program

# 仅 Profile 特定 kernel
ncu --kernel-name-base function --kernel-name mySaxpy ./my_program

# Skip 前 N 个 kernel
ncu --launch-skip 100 ./my_program

# 只 Profile N 个 kernel
ncu --launch-count 10 ./my_program
```

### 3. Section 详解

NCU 使用 "sections" 组织相关指标：

```bash
# 查看所有可用 sections
ncu --query-sections

# 常用 sections
--section SpeedOfLight          # 总体性能概览
--section MemoryWorkloadAnalysis # 内存工作负载
--section ComputeWorkloadAnalysis # 计算工作负载
--section SchedulerStats        # 调度器统计
--section WarpStateStats        # Warp 状态统计
--section Occupancy             # 占用率分析
--section SourceCounters        # 源代码级计数器
```

### 4. Metrics 详解

```bash
# 查看所有可用 metrics
ncu --query-metrics

# 重要 metrics 示例
--metrics sm__throughput.avg.pct_of_peak_sustained_elapsed  # SM 吞吐量
--metrics dram__throughput.avg.pct_of_peak_sustained_elapsed # DRAM 带宽利用率
--metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed # L1/Tex 缓存吞吐量
--metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum # FP Add 指令数
```

---

## 核心指标解析

### 1. Speed of Light (SOL)

这是最重要的起点指标，显示各个子系统的利用率。

```bash
ncu --section SpeedOfLight ./my_program
```

**关键指标：**

| 指标 | 含义 | 目标 |
|------|------|------|
| **SM Throughput** | SM 计算单元利用率 | > 80% (compute-bound) |
| **Memory Throughput** | 内存带宽利用率 | > 80% (memory-bound) |
| **L1/TEX Hit Rate** | L1 缓存命中率 | > 90% (理想) |
| **L2 Hit Rate** | L2 缓存命中率 | > 70% (理想) |

**解读示例：**
```
SM Throughput:       25%   ← 计算单元利用率低
Memory Throughput:   85%   ← 内存带宽接近饱和
```
**结论：** Memory-bound，需要优化内存访问

### 2. Memory Workload Analysis

```bash
ncu --section MemoryWorkloadAnalysis ./my_program
```

**关键指标：**

#### Global Memory
```
Metric: dram__bytes_read.sum                    # 从 DRAM 读取的字节数
Metric: dram__bytes_write.sum                   # 写入 DRAM 的字节数
Metric: dram__throughput.avg.pct_of_peak        # DRAM 带宽利用率
```

#### L2 Cache
```
Metric: lts__t_sectors_op_read.sum              # L2 读扇区数
Metric: lts__t_sectors_op_write.sum             # L2 写扇区数
Metric: lts__t_sectors_op_red.sum               # L2 原子操作扇区数
```

#### L1/Tex Cache
```
Metric: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  # L1 全局加载
Metric: l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum  # L1 全局存储
```

#### Shared Memory
```
Metric: l1tex__data_pipe_lsu_wavefronts_mem_shared.sum  # Shared memory 访问
Metric: l1tex__data_bank_conflicts_pipe_lsu.sum         # Bank conflicts
```

**计算实际带宽：**
```python
# 理论带宽
theoretical_bandwidth = DRAM_bandwidth  # 例如 A100: 1555 GB/s

# 实际带宽
actual_bandwidth = (bytes_read + bytes_write) / kernel_duration

# 带宽利用率
bandwidth_utilization = actual_bandwidth / theoretical_bandwidth * 100
```

### 3. Compute Workload Analysis

```bash
ncu --section ComputeWorkloadAnalysis ./my_program
```

**关键指标：**

```
# 指令统计
smsp__inst_executed.sum                         # 执行的指令总数
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum  # FP32 加法
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum  # FP32 乘法
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum  # FP32 FMA

# Warp 效率
smsp__thread_inst_executed_per_inst_executed.ratio   # 平均每指令执行的线程数
                                                      # 理想值: 32（无 divergence）
```

**计算 FLOPs：**
```python
# FP32 FLOPs
flops = (fadd_ops + fmul_ops + 2 * ffma_ops)

# 计算吞吐量
compute_throughput = flops / kernel_duration  # FLOPS

# vs 理论峰值
peak_flops = GPU_peak_TFLOPS * 1e12
compute_efficiency = compute_throughput / peak_flops * 100
```

### 4. Occupancy

```bash
ncu --section Occupancy ./my_program
```

**关键指标：**

```
Occupancy (Theory):  50%     # 理论占用率（基于资源限制）
Occupancy (Achieved): 45%    # 实际占用率
```

**限制因素：**
- Registers per thread
- Shared memory per block
- Threads per block
- Blocks per SM

**占用率计算器：**
```bash
# 使用 CUDA Occupancy Calculator
ncu --print-summary per-kernel

# 或在报告中查看 "Occupancy" section
```

### 5. Warp State Statistics

```bash
ncu --section WarpStateStats ./my_program
```

**Warp 状态分布：**
```
Active:         60%    # 正在执行指令
Stalled:        40%    # 停顿
  - Memory:     25%    # 等待内存
  - Sync:       10%    # 等待同步
  - Math:       3%     # 等待计算单元
  - Other:      2%
```

**优化目标：**
- 降低 Stalled 比例
- 如果 Memory Stall 高 → 优化内存访问
- 如果 Sync Stall 高 → 减少 `__syncthreads()`

---

## 高级分析技巧

### 1. Roofline Analysis

```bash
ncu --section SpeedOfLight_RooflineChart ./my_program
```

**Roofline Model 解释：**

```
        Compute Roof (Peak FLOPS)
             |
Performance  |         *  (Your kernel)
             |       /
             |     /
             |   /
             | /  Memory Roof (Peak Bandwidth)
             |/_________________________
                Arithmetic Intensity
```

**读取 Roofline：**
1. 如果 kernel 在 Memory Roof 下 → Memory-bound
2. 如果 kernel 在 Compute Roof 下 → Compute-bound
3. 距离 Roof 的距离 = 优化空间

### 2. Source-Level Analysis

```bash
# 需要编译时包含调试信息
nvcc -lineinfo kernel.cu -o my_program

# Profile 并启用源代码分析
ncu --section SourceCounters ./my_program

# 在 GUI 中查看
ncu-ui profile_result.ncu-rep
```

**在 GUI 中：**
- Source 页面显示源代码
- 每行旁边显示性能指标
- 可以看到哪行代码最耗时

### 3. 对比分析

```bash
# Profile 两个版本
ncu -o baseline ./program_v1
ncu -o optimized ./program_v2

# 在 GUI 中对比
ncu-ui baseline.ncu-rep optimized.ncu-rep

# 命令行对比
ncu --import baseline.ncu-rep --import optimized.ncu-rep
```

### 4. Kernel Replay

NCU 使用 kernel replay 技术：
- 首先运行程序记录 kernel
- 然后 replay kernel 并收集指标
- 允许收集大量指标而不减慢程序

**注意事项：**
- Kernel 不能有副作用依赖
- 某些复杂 kernel 可能无法 replay

```bash
# 禁用 kernel replay（如果有问题）
ncu --replay-mode application ./my_program

# 启用 kernel replay（默认）
ncu --replay-mode kernel ./my_program
```

### 5. 规则和建议

```bash
# NCU 提供自动化的性能建议
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./my_program
```

在报告中查找 "Details" section，会有：
- 🔴 High priority issues
- 🟡 Medium priority suggestions
- 🟢 Informational notes

### 6. Python API

```python
import pynvml
import subprocess

def profile_kernel(program, kernel_name):
    """使用 NCU Python API"""
    cmd = [
        'ncu',
        '--metrics', 'sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed',
        '--kernel-name', kernel_name,
        '--csv',
        program
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    # 解析 CSV 输出
    # ... 处理数据
```

---

## 实战案例

### 案例 1：诊断 Memory-Bound Kernel

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./matrix_transpose
```

**NCU 输出：**
```
Speed of Light:
  SM Throughput:     15%    ← 很低
  Memory Throughput: 95%    ← 接近峰值

Memory Workload:
  Global Load:       80 GB
  Global Store:      80 GB
  Duration:          10 ms

Actual Bandwidth:    16 TB/s  (理论: 1.5 TB/s for A100)
  → 计算错误！应该是 16 GB/s
```

**分析：**
- Memory-bound（内存带宽瓶颈）
- 实际带宽远低于理论值
- 可能是非合并访问

**进一步分析：**
```bash
ncu --section MemoryWorkloadAnalysis_Chart ./matrix_transpose
```

查看：
- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld` > 1
  - 表示非合并访问

**解决方案：**
- 使用 Shared Memory
- 优化访问模式

### 案例 2：诊断 Low Occupancy

```bash
ncu --section Occupancy ./my_kernel
```

**NCU 输出：**
```
Occupancy:
  Theoretical:  25%
  Achieved:     23%

Limiting Factors:
  ✓ Blocks per SM: 4
  ⚠ Registers per Thread: 128 (Limit: 64)
  ✓ Shared Memory per Block: 16 KB
```

**分析：**
- 寄存器使用过多限制了占用率

**解决方案：**
```cpp
// 使用 launch bounds 限制寄存器
__global__ void __launch_bounds__(256, 4)
my_kernel(...) {
    // kernel code
}
```

**或者编译时：**
```bash
nvcc -maxrregcount=64 kernel.cu
```

### 案例 3：诊断 Bank Conflicts

```bash
ncu --section MemoryWorkloadAnalysis ./my_kernel
```

**查看指标：**
```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum: 10,000
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum: 8,000
```

**高 bank conflicts！**

**定位问题：**
```bash
# 启用源代码分析
ncu --section SourceCounters -o profile ./my_kernel
ncu-ui profile.ncu-rep
```

在源代码视图中找到 bank conflict 热点。

**解决方案：**
```cpp
// 添加 padding 避免 bank conflicts
__shared__ float shared[32][33];  // +1 padding
```

### 案例 4：Warp Divergence

```bash
ncu --section WarpStateStats --section ComputeWorkloadAnalysis ./my_kernel
```

**查看：**
```
smsp__thread_inst_executed_per_inst_executed.ratio: 16
  → 平均每条指令只有 16 个线程执行（理想是 32）
  → 有严重的 warp divergence
```

**定位代码：**
```cpp
__global__ void divergent_kernel(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 问题代码：造成 divergence
    if (idx % 2 == 0) {
        data[idx] = expensive_computation();
    } else {
        data[idx] = cheap_computation();
    }
}
```

**解决方案：**
- 重组数据使条件分支在 warp 内一致
- 或者使用 predication（谓词执行）

---

## 性能优化工作流

### 标准工作流

```
1. 初步 Profile
   ↓
   ncu --section SpeedOfLight ./program
   ↓
2. 识别瓶颈
   ↓
   Memory-bound? → Step 3
   Compute-bound? → Step 4
   Low Occupancy? → Step 5
   ↓
3. Memory 优化
   ↓
   ncu --section MemoryWorkloadAnalysis ./program
   - 检查带宽利用率
   - 检查缓存命中率
   - 检查 bank conflicts
   ↓
4. Compute 优化
   ↓
   ncu --section ComputeWorkloadAnalysis ./program
   - 检查指令吞吐量
   - 检查 warp divergence
   - 检查 ILP
   ↓
5. Occupancy 优化
   ↓
   ncu --section Occupancy ./program
   - 调整 block size
   - 减少寄存器使用
   - 减少 shared memory
   ↓
6. 验证改进
   ↓
   对比前后性能
   ↓
7. 迭代优化
```

### 快速检查脚本

```bash
#!/bin/bash
# profile_quick.sh

PROGRAM=$1
KERNEL=$2

echo "=== Quick Profile ==="
echo ""

echo "1. Speed of Light:"
ncu --section SpeedOfLight \
    --kernel-name $KERNEL \
    $PROGRAM 2>&1 | grep -A 20 "SOL"

echo ""
echo "2. Memory Throughput:"
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name $KERNEL \
    $PROGRAM

echo ""
echo "3. Occupancy:"
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --kernel-name $KERNEL \
    $PROGRAM

echo ""
echo "4. Bank Conflicts:"
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu.sum \
    --kernel-name $KERNEL \
    $PROGRAM
```

**使用：**
```bash
chmod +x profile_quick.sh
./profile_quick.sh ./my_program my_kernel_name
```

### 持续监控

```python
#!/usr/bin/env python3
"""持续性能监控脚本"""

import subprocess
import json
import time

def profile_kernel(program, kernel):
    cmd = [
        'ncu',
        '--metrics',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed,'
        'dram__throughput.avg.pct_of_peak_sustained_elapsed,'
        'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.pct',
        '--kernel-name', kernel,
        '--csv',
        program
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_csv(result.stdout)

def parse_csv(csv_output):
    # 解析 NCU CSV 输出
    lines = csv_output.strip().split('\n')
    # ... 解析逻辑
    return metrics

def main():
    metrics_history = []

    while True:
        metrics = profile_kernel('./my_program', 'my_kernel')
        metrics_history.append(metrics)

        # 检测性能回归
        if len(metrics_history) > 1:
            if metrics['sm_throughput'] < metrics_history[-2]['sm_throughput'] * 0.9:
                print("⚠️  Performance regression detected!")

        time.sleep(60)  # 每分钟检查一次

if __name__ == '__main__':
    main()
```

---

## NCU 命令速查表

```bash
# 基础 profile
ncu ./program                           # 基本 profile
ncu -o output ./program                 # 保存结果
ncu --set full -o output ./program      # 完整指标集

# Kernel 选择
ncu --kernel-name matmul ./program      # 特定 kernel
ncu --kernel-id ::matmul:2 ./program    # 第 2 次调用
ncu --launch-skip 10 ./program          # 跳过前 10 个
ncu --launch-count 5 ./program          # 只 profile 5 个

# Sections
ncu --section SpeedOfLight              # 性能概览
ncu --section MemoryWorkloadAnalysis    # 内存分析
ncu --section ComputeWorkloadAnalysis   # 计算分析
ncu --section Occupancy                 # 占用率
ncu --section WarpStateStats            # Warp 状态

# Metrics
ncu --metrics <metric_name>             # 特定指标
ncu --query-metrics                     # 列出所有指标

# 导出
ncu --csv                               # CSV 格式
ncu --page raw                          # 原始数据
ncu --export output.csv                 # 导出到文件

# GUI
ncu-ui output.ncu-rep                   # 打开报告
```

---

## 故障排除

### 问题 1：权限错误
```
ERR_NVGPUCTRPERM - The user does not have permission to profile
```

**解决：**
```bash
sudo su
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
update-initramfs -u
reboot
```

### 问题 2：Kernel Replay 失败
```
Failed to replay kernel
```

**解决：**
```bash
# 使用 application replay
ncu --replay-mode application ./program
```

### 问题 3：指标收集失败
```
Metric not available on this device
```

**解决：**
- 检查 GPU 架构是否支持该指标
- 使用 `--query-metrics` 查看可用指标

---

## 参考资源

- [Nsight Compute 官方文档](https://docs.nvidia.com/nsight-compute/)
- [Nsight Compute CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
- [Metrics Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

---

## 下一步

- 学习 [Nsight Systems](nsight_systems.md) 进行系统级分析
- 查看 [全局优化视角](../05_global_optimization/global_perspective.md)
- 实践完整的性能优化流程
