# 自动化分析工具

这套工具用于自动化 GPU 性能分析、对比和可视化。

## 工具列表

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| `auto_profile.py` | 自动 NCU 分析 + 瓶颈诊断 | 快速诊断单个 kernel 的性能问题 |
| `compare_versions.py` | 多版本性能对比 | 对比优化前后的性能差异 |
| `visualize.py` | 性能可视化图表 | 生成论文/报告用的图表 |
| `roofline.py` | Roofline 模型分析 | 判断 Memory-bound vs Compute-bound |

## 快速开始

### 1. 安装依赖

```bash
pip install matplotlib numpy
```

### 2. 单个 Kernel 分析

```bash
# 自动运行 NCU 并诊断瓶颈
python auto_profile.py ./v0_naive

# 输出示例:
# ======================================================================
# NCU Profile Result
# ======================================================================
#
# Binary: ./v0_naive
# Kernel: vector_add_naive
#
# Performance:
#   Duration: 12.4500 ms
#   DRAM Bandwidth: 61.40 GB/s
#
# Speed of Light:
#   SM Throughput: 12.3% 🔴
#   Memory Throughput: 89.2% 🔴
#
# Memory Analysis:
#   Sectors per Request: 32.00 🔴
#   L2 Hit Rate: 2.1%
#
# Bottleneck: Memory-bound
#
# Optimization Suggestions:
#   • ❌ 非合并访问 (sectors_per_request=32.00)
#      → 修复为连续访问 (预期 8-10x 提升)
#      → 参考: benchmarks/vector_ops/vector_add/v0→v1
```

### 3. 多版本对比

```bash
cd ../benchmarks/vector_ops/vector_add
make all  # 编译所有版本

# 对比所有版本
python ../../../tools/compare_versions.py v0_naive v1_coalesced v2_vectorized v4_optimized

# 导出结果
python ../../../tools/compare_versions.py v*.cu --json=results.json --csv=results.csv
```

输出示例:
```
============================================================================
Performance Comparison Table
============================================================================
Version                    Time (ms)     Speedup  BW (GB/s)   BW Ratio  ...
----------------------------------------------------------------------------
v0_naive                      12.4500        1.00x      61.40       1.00x
v1_coalesced                   1.5200        8.19x     502.60       8.19x
v2_vectorized                  0.7100       17.54x    1077.50      17.54x
v4_optimized                   0.5400       23.06x    1416.70      23.06x
============================================================================

Summary:
  Fastest: v4_optimized (0.5400 ms)
  Slowest: v0_naive (12.4500 ms)
  Total improvement: 23.06x
```

### 4. 性能可视化

```bash
# 从对比结果生成图表
python visualize.py results.json --output=charts.png

# 生成包含以下图表:
# - 加速比柱状图
# - 带宽利用率对比
# - 优化时间线
# - NCU 指标雷达图
# - 瓶颈类型分布
# - 性能汇总表
```

### 5. Roofline 分析

```bash
# 手动输入参数
python roofline.py --flops=67108864 --bytes=805306368 --time=12.45 --plot

# 输出:
# ======================================================================
# Roofline Model Analysis
# ======================================================================
#
# GPU: NVIDIA A100
#   Peak Compute: 19500.0 GFLOPS
#   Peak Bandwidth: 1555.0 GB/s
#   Ridge Point: 12.54 FLOPS/Byte
#
# Kernel Performance:
#   Arithmetic Intensity (AI): 0.083 FLOPS/Byte
#   Achieved Performance: 5.39 GFLOPS
#   Achieved Bandwidth: 64.68 GB/s
#
# Analysis:
#   Bottleneck: Memory-bound
#   Performance Ceiling: 129.41 GFLOPS
#   Efficiency: 4.2%
#
# Optimization Suggestions:
#   • 🔴 Kernel 是 Memory-bound (AI=0.083 < Ridge Point=12.54)
#   • 内存访问效率低，优化方向:
#        1. 检查 memory coalescing (NCU: sectors_per_request)
#        2. 使用 float4 向量化访问
#        3. 考虑 Shared Memory (如果有数据重用)

# 生成 roofline.png 图表
```

## 完整工作流示例

### Vector Add 优化案例

```bash
# 1. 编译所有版本
cd benchmarks/vector_ops/vector_add
make all

# 2. 快速诊断 baseline
python ../../../tools/auto_profile.py ./v0_naive
# 发现: sectors_per_request=32.0 → 非合并访问

# 3. 修复后验证 (v1_coalesced)
python ../../../tools/auto_profile.py ./v1_coalesced
# 验证: sectors_per_request=1.0 ✅

# 4. 对比所有版本
python ../../../tools/compare_versions.py \
  v0_naive v1_coalesced v2_vectorized v3_shared_tiling v4_optimized \
  --json=comparison.json \
  --csv=comparison.csv

# 5. 生成可视化报告
python ../../../tools/visualize.py comparison.json --output=report.png

# 6. Roofline 分析 (判断是否值得继续优化)
python ../../../tools/roofline.py \
  --flops=67108864 \
  --bytes=805306368 \
  --time=0.54 \
  --plot \
  --output=roofline_v4.png
```

结果:
- v0 → v1: 修复 coalescing (8x 提升)
- v1 → v2: 向量化 (2x 提升)
- v2 → v3: Shared Memory (❌ 性能下降 - 反面教材)
- v2 → v4: Fine-tuning (1.5x 提升)
- **总提升: 23x**
- **Roofline 显示已达 91% 带宽效率 → 停止优化内核**

## 工具详细说明

### auto_profile.py

**功能**: 自动化 NCU 分析

**输入**:
- 二进制文件路径
- (可选) 命令行参数

**输出**:
- NCU 关键指标
- 瓶颈诊断 (Memory/Compute/Launch-bound)
- 优化建议
- JSON 结果文件

**原理**:
1. 调用 `ncu --metrics <key_metrics> <binary>`
2. 解析 CSV 输出提取指标
3. 应用四象限诊断矩阵:
   ```
   SM% > 60, Mem% < 40  → Compute-bound
   SM% < 40, Mem% > 60  → Memory-bound
   SM% < 40, Mem% < 40  → Launch-bound
   SM% > 60, Mem% > 60  → Balanced (已优化)
   ```
4. 根据诊断结果生成具体建议

**关键指标**:
- `gpu__time_duration.avg`: 执行时间
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: SM 利用率
- `gpu__compute_memory.avg.pct_of_peak_sustained_elapsed`: 显存利用率
- `l1tex__average_t_sectors_per_request`: 访问合并度 (期望 ~1.0)
- `lts__t_sector_hit_rate.pct`: L2 命中率
- `sm__warps_active.avg.pct_of_peak_sustained_active`: Occupancy

### compare_versions.py

**功能**: 多版本性能对比

**输入**:
- 多个二进制文件
- (可选) Baseline 版本

**输出**:
- 对比表格 (Time, Speedup, Bandwidth, Bottleneck)
- 性能回归检测
- JSON/CSV 导出

**使用技巧**:
```bash
# 使用通配符
python compare_versions.py v*.cu

# 指定 baseline
python compare_versions.py v*.cu --baseline=v1_coalesced

# 详细模式
python compare_versions.py v*.cu --detailed
```

**性能回归检测**:
- 自动检测相比前一版本变慢 >5% 的情况
- 标记为 "⚠️ Slower"

### visualize.py

**功能**: 生成性能可视化图表

**输入**:
- `compare_versions.py` 生成的 JSON 文件

**输出**:
- 综合图表 (PNG/PDF/SVG)
- 包含 6 个子图:
  1. 加速比柱状图
  2. 带宽利用率对比
  3. 优化时间线
  4. 瓶颈类型分布饼图
  5. NCU 指标雷达图 (Baseline vs Optimized)
  6. 性能汇总表

**依赖**: `matplotlib`, `numpy`

**输出格式**:
```bash
# PNG (默认)
python visualize.py results.json

# PDF (论文质量)
python visualize.py results.json --format=pdf

# SVG (矢量图)
python visualize.py results.json --format=svg
```

### roofline.py

**功能**: Roofline 模型分析

**输入**:
- FLOPS (浮点操作数)
- Bytes (内存访问字节数)
- Time (执行时间 ms)

**输出**:
- Arithmetic Intensity (AI)
- 瓶颈判断 (Memory-bound vs Compute-bound)
- 性能上界 (Ceiling)
- 效率 (Actual / Ceiling)
- Roofline 图表

**支持 GPU**:
- A100 (FP32: 19.5 TFLOPS, BW: 1555 GB/s)
- V100 (FP32: 15.7 TFLOPS, BW: 900 GB/s)
- RTX 3090 (FP32: 35.6 TFLOPS, BW: 936 GB/s)

**计算公式**:
```python
AI = FLOPS / Bytes

Ridge Point = Peak Compute / Peak Bandwidth

if AI < Ridge Point:
    Bottleneck = "Memory-bound"
    Ceiling = AI × Peak Bandwidth
else:
    Bottleneck = "Compute-bound"
    Ceiling = Peak Compute

Efficiency = Achieved Performance / Ceiling
```

**示例 - Vector Add**:
```python
# c[i] = a[i] + b[i], N = 64M
FLOPS = N = 67108864
Bytes = 3 × N × 4 = 805306368  # Read a, b, Write c
AI = 67M / 805M = 0.083 FLOPS/Byte

# A100 Ridge Point = 12.54 FLOPS/Byte
# 0.083 << 12.54 → Memory-bound!

# Ceiling = 0.083 × 1555 GB/s = 129 GFLOPS
# v4 Achieved = 124 GFLOPS
# Efficiency = 96% → 已达极限!
```

## 高级用法

### 批量分析多个 benchmark

```bash
#!/bin/bash
# analyze_all.sh

BENCHMARKS=(
    "benchmarks/vector_ops/vector_add/v4_optimized"
    "benchmarks/matrix_ops/gemm/gemm_v7"
    "benchmarks/reduction/reduce_v5"
)

for bench in "${BENCHMARKS[@]}"; do
    echo "Analyzing $bench..."
    python tools/auto_profile.py $bench
done

# 生成汇总报告
python tools/generate_report.py *.json
```

### 自定义 NCU 指标

修改 `auto_profile.py`:

```python
KEY_METRICS = [
    "gpu__time_duration.avg",
    # ... 默认指标
    # 添加自定义指标
    "smsp__inst_executed_pipe_tensor",  # Tensor Core 使用
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld",  # 向量化程度
]
```

### 集成到 CI/CD

```yaml
# .github/workflows/performance_test.yml
name: Performance Regression Test

on: [push, pull_request]

jobs:
  perf-test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2

      - name: Build kernels
        run: make all

      - name: Profile performance
        run: |
          python tools/compare_versions.py v*.cu --json=results.json

      - name: Check regression
        run: |
          python tools/check_regression.py results.json --threshold=0.95
```

## 故障排除

### NCU 权限问题

```bash
# 需要 sudo 运行 NCU
sudo python auto_profile.py ./kernel

# 或者设置权限
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo reboot
```

### matplotlib 后端问题

如果出现显示错误:
```python
# 在脚本开头添加
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
```

### NCU CSV 解析失败

检查 NCU 版本:
```bash
ncu --version
# 建议 2021.1 或更高版本
```

## 相关文档

- [NCU UI 使用指南](../04_performance_analysis/ncu_ui_guide.md) - NCU 界面详解
- [NCU 专家级分析](../04_performance_analysis/ncu_expert_analysis.md) - 深入指标解读
- [全局优化框架](../frameworks/global_optimization_framework.md) - 优化方法论
- [Vector Add Benchmark](../benchmarks/vector_ops/vector_add/) - 完整示例

## 贡献

欢迎提交 Issue 和 PR:
- 添加新的 GPU 型号支持
- 改进瓶颈诊断算法
- 添加更多可视化图表
- 优化 NCU 指标解析
