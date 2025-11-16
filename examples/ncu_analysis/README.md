# NCU 分析示例

这个目录包含了使用 NCU 进行性能分析的实用脚本和示例。

## 文件说明

- `analyze_ncu.py` - 自动分析 NCU CSV 输出的 Python 脚本
- `example_analysis.sh` - NCU 分析工作流示例脚本

## 使用方法

### 1. 收集 NCU 数据

```bash
# 方法 1：导出为 CSV（用于自动分析）
ncu --csv --page raw ./your_program > ncu_output.csv

# 方法 2：导出为 .ncu-rep（用于 GUI 查看）
ncu --set full -o profile_result ./your_program

# 方法 3：只收集特定 sections
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --csv --page raw \
    ./your_program > ncu_output.csv
```

### 2. 自动分析

```bash
# 使用 Python 脚本分析
python analyze_ncu.py ncu_output.csv
```

### 3. 查看详细报告

```bash
# 打开 GUI
ncu-ui profile_result.ncu-rep
```

## 分析工作流示例

### 场景 1：首次分析一个 kernel

```bash
# Step 1: 快速诊断（只用 SpeedOfLight）
ncu --section SpeedOfLight ./program

# 查看输出，判断瓶颈类型
# - SM Throughput > 80%? → Compute-bound
# - Memory Throughput > 80%? → Memory-bound

# Step 2: 根据瓶颈类型深入分析

# 如果是 Memory-bound:
ncu --section MemoryWorkloadAnalysis \
    --csv --page raw \
    ./program > memory_analysis.csv

python analyze_ncu.py memory_analysis.csv

# 如果是 Compute-bound:
ncu --section ComputeWorkloadAnalysis \
    --csv --page raw \
    ./program > compute_analysis.csv

python analyze_ncu.py compute_analysis.csv

# Step 3: 完整分析（如果需要）
ncu --set full -o full_profile ./program
ncu-ui full_profile.ncu-rep
```

### 场景 2：优化前后对比

```bash
# 优化前
ncu --set full -o baseline ./program_old

# 优化后
ncu --set full -o optimized ./program_new

# GUI 对比
ncu-ui baseline.ncu-rep optimized.ncu-rep

# 命令行对比关键指标
ncu --import baseline.ncu-rep \
    --import optimized.ncu-rep \
    --page raw \
    --csv > comparison.csv
```

### 场景 3：批量分析多个 kernels

```bash
#!/bin/bash
# 分析程序中的所有 kernels

PROGRAM="./my_program"

# 列出所有 kernels
ncu --query-kernel-names $PROGRAM

# 对每个 kernel 进行分析
for kernel in $(ncu --query-kernel-names $PROGRAM); do
    echo "Analyzing $kernel..."
    ncu --kernel-name $kernel \
        --section SpeedOfLight \
        --csv --page raw \
        $PROGRAM > "${kernel}_analysis.csv"

    python analyze_ncu.py "${kernel}_analysis.csv"
done
```

## 输出示例

运行 `python analyze_ncu.py ncu_output.csv` 后的输出：

```
================================================================================
NCU 性能分析报告
================================================================================
Kernel: vectorAdd(float*, float*, float*, int)

================================================================================
第一步：Speed of Light 分析
================================================================================
SM Throughput:            12.34%
Memory Throughput:        87.56%

结论: Memory-bound ✓
内存带宽接近饱和，计算单元空闲

================================================================================
第二步：Memory 深入分析
================================================================================

📊 DRAM 带宽分析:
  DRAM Throughput:        85.32%
  读取字节数:              400.00 GB
  写入字节数:              200.00 GB
  实际带宽:              1320.45 GB/s
  理论带宽:              1555.00 GB/s (A100)
  带宽利用率:              84.92%
  ✓ 带宽利用率高，接近硬件极限

📊 L2 Cache 分析:
  L2 命中率:               25.67%
  ❌ L2 命中率低，数据几乎没有复用
     建议：算子融合、Tiling、增加数据复用

📊 L1/TEX Cache 分析:
  L1 命中率:               88.34%
  ⚠️  L1 命中率一般

📊 Coalesced Access 分析:
  加载 Coalesced 程度:      1.02 (理想值 = 1.0)
  存储 Coalesced 程度:      1.01 (理想值 = 1.0)
  ✓ 内存访问模式良好，接近完美合并

📊 Shared Memory Bank Conflicts:
  Bank Conflicts:                 0
  Shared Accesses:           500000
  Conflict 率:                0.00%
  ✓ 几乎无 bank conflicts

================================================================================
💡 优化建议总结
================================================================================

✅ 优先级 1（最重要）：
  1. 算子融合：将多个 kernel 合并，减少内存访问
  2. 使用 Shared Memory：缓存频繁访问的数据

✅ 优先级 2：
  3. 向量化访问：使用 float4 提高带宽利用率
  4. 优化访问模式：确保 coalesced access

✅ 优先级 3：
  5. 提高缓存命中率：Tiling、增加数据复用
  6. 消除 Bank Conflicts：添加 padding
```

## 常见问题排查

### Q1: CSV 文件为空或格式错误

```bash
# 确保使用正确的参数
ncu --csv --page raw ./program > output.csv

# 而不是
ncu --csv ./program > output.csv  # ❌ 缺少 --page raw
```

### Q2: 某些指标缺失

```bash
# 使用 --set full 收集所有指标
ncu --set full --csv --page raw ./program > output.csv
```

### Q3: 需要管理员权限

```bash
# 临时获取权限
sudo ncu --csv --page raw ./program > output.csv

# 或永久设置（不推荐生产环境）
sudo su
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
update-initramfs -u
reboot
```

## 参考

- [NCU 工作流指南](../../04_performance_analysis/ncu_workflow.md)
- [NCU 指标参考](../../04_performance_analysis/ncu_metrics_reference.md)
- [NCU 官方文档](https://docs.nvidia.com/nsight-compute/)
