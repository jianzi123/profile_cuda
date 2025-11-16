# NCU 关键指标速查手册

## 目录
1. [Speed of Light 指标](#speed-of-light-指标)
2. [Memory 指标](#memory-指标)
3. [Compute 指标](#compute-指标)
4. [Occupancy 指标](#occupancy-指标)
5. [Warp State 指标](#warp-state-指标)
6. [指标组合判断](#指标组合判断)

---

## Speed of Light 指标

### 整体性能指标

| 指标名称 | 含义 | 理想值 | 问题诊断 |
|---------|------|--------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 计算吞吐量 | > 80% (compute-bound) | < 40% 计算单元闲置 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 整体内存吞吐量 | > 80% (memory-bound) | < 60% 内存不是瓶颈 |
| `gpu__time_duration.sum` | Kernel 执行时间 | 越低越好 | 用于计算带宽/吞吐量 |

### 内存系统吞吐量

| 指标名称 | 含义 | 理想值 | 问题诊断 |
|---------|------|--------|---------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率 | > 80% | DRAM 是瓶颈 |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 Cache 吞吐量 | > 70% | L2 压力大 |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1/TEX Cache 吞吐量 | > 60% | L1 利用率高 |

### 计算单元吞吐量

| 指标名称 | 含义 | 理想值 | 问题诊断 |
|---------|------|--------|---------|
| `sm__inst_executed.avg.pct_of_peak_sustained_active` | 指令执行率 | > 70% | 计算利用率低 |
| `smsp__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active` | FP32 流水线利用率 | > 80% (FP32 kernel) | FP32 计算不饱和 |
| `smsp__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active` | FP16 流水线利用率 | > 80% (FP16 kernel) | FP16 计算不饱和 |
| `smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` | Tensor Core 利用率 | > 80% (TC kernel) | Tensor Core 未充分利用 |

---

## Memory 指标

### DRAM (Global Memory) 指标

| 指标名称 | 单位 | 含义 | 如何使用 |
|---------|------|------|---------|
| `dram__bytes_read.sum` | bytes | DRAM 读取总字节数 | 计算读带宽 |
| `dram__bytes_write.sum` | bytes | DRAM 写入总字节数 | 计算写带宽 |
| `dram__sectors_read.sum` | sectors | DRAM 读取扇区数 | 32 bytes/sector |
| `dram__sectors_write.sum` | sectors | DRAM 写入扇区数 | 32 bytes/sector |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % | DRAM 带宽利用率 | > 80% 说明达到瓶颈 |

**带宽计算公式**：
```python
total_bytes = dram__bytes_read.sum + dram__bytes_write.sum
duration_sec = gpu__time_duration.sum / 1e9  # ns to sec
actual_bandwidth_GBps = total_bytes / duration_sec / 1e9

# 对于 A100: theoretical_bandwidth = 1555 GB/s
utilization_pct = (actual_bandwidth_GBps / theoretical_bandwidth) * 100
```

### L2 Cache 指标

| 指标名称 | 单位 | 含义 | 理想值 |
|---------|------|------|-------|
| `lts__t_sectors.sum` | sectors | L2 访问总扇区数 | - |
| `lts__t_sectors_op_read.sum` | sectors | L2 读扇区数 | - |
| `lts__t_sectors_op_write.sum` | sectors | L2 写扇区数 | - |
| `lts__t_sector_hit_rate.pct` | % | **L2 命中率** | **> 70%** |
| `lts__t_bytes.sum` | bytes | L2 访问总字节数 | - |

**L2 命中率解读**：
```
> 90%: 优秀的数据复用
70-90%: 良好的缓存利用
50-70%: 一般，有改进空间
< 50%: 差，数据几乎没有复用
```

### L1/TEX Cache 指标

| 指标名称 | 单位 | 含义 | 理想值 |
|---------|------|------|-------|
| `l1tex__t_sectors.sum` | sectors | L1 访问总扇区数 | - |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | sectors | L1 全局加载扇区数 | - |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | sectors | L1 全局存储扇区数 | - |
| `l1tex__t_sector_hit_rate.pct` | % | **L1 命中率** | **> 90%** |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | ratio | **每次加载请求的平均扇区数** | **= 1.0 (完美 coalesced)** |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio` | ratio | **每次存储请求的平均扇区数** | **= 1.0 (完美 coalesced)** |

**Coalesced Access 判断**：
```
= 1.0:  ✅ 完美合并访问，无带宽浪费
= 1.5:  ⚠️  浪费 33% 带宽
= 2.0:  ❌ 浪费 50% 带宽 (跨步访问)
= 4.0:  ❌ 浪费 75% 带宽 (严重的非合并访问)
> 4.0:  ❌ 随机访问或严重未对齐
```

### Shared Memory 指标

| 指标名称 | 单位 | 含义 | 理想值 |
|---------|------|------|-------|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | wavefronts | Shared memory 访问次数 | - |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | conflicts | Shared memory 加载 bank conflicts | **= 0** |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | conflicts | Shared memory 存储 bank conflicts | **= 0** |
| `l1tex__data_bank_conflicts_pipe_lsu.sum` | conflicts | 总 bank conflicts | **= 0** |

**Bank Conflict 率计算**：
```python
shared_accesses = l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
bank_conflicts = l1tex__data_bank_conflicts_pipe_lsu.sum

conflict_rate_pct = (bank_conflicts / shared_accesses) * 100

# 解读
< 1%:   ✅ 优秀
1-10%:  ⚠️  可接受，但有改进空间
10-30%: ❌ 差，需要优化
> 30%:  ❌ 严重问题，严重降低性能
```

---

## Compute 指标

### 指令统计

| 指标名称 | 单位 | 含义 | 用途 |
|---------|------|------|-----|
| `smsp__inst_executed.sum` | instructions | 执行的总指令数 | 统计指令量 |
| `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` | instructions | FP32 加法指令 | 计算 FP32 FLOPs |
| `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` | instructions | FP32 乘法指令 | 计算 FP32 FLOPs |
| `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | instructions | FP32 FMA 指令 | 计算 FP32 FLOPs (×2) |
| `smsp__sass_thread_inst_executed_op_hadd_pred_on.sum` | instructions | FP16 加法指令 | 计算 FP16 FLOPs |
| `smsp__sass_thread_inst_executed_op_hmul_pred_on.sum` | instructions | FP16 乘法指令 | 计算 FP16 FLOPs |
| `smsp__sass_thread_inst_executed_op_hfma_pred_on.sum` | instructions | FP16 FMA 指令 | 计算 FP16 FLOPs (×2) |
| `smsp__sass_thread_inst_executed_op_dadd_pred_on.sum` | instructions | FP64 加法指令 | 计算 FP64 FLOPs |
| `smsp__sass_thread_inst_executed_op_dmul_pred_on.sum` | instructions | FP64 乘法指令 | 计算 FP64 FLOPs |
| `smsp__sass_thread_inst_executed_op_dfma_pred_on.sum` | instructions | FP64 FMA 指令 | 计算 FP64 FLOPs (×2) |

**FLOPs 计算**：
```python
# FP32 FLOPs
fp32_flops = (
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum +
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum +
    2 * smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
)

# 计算吞吐量
duration_sec = gpu__time_duration.sum / 1e9
fp32_tflops = fp32_flops / duration_sec / 1e12

# 与理论峰值对比 (A100 = 19.5 TFLOPS FP32)
efficiency_pct = (fp32_tflops / 19.5) * 100
```

### Warp Efficiency

| 指标名称 | 单位 | 含义 | 理想值 |
|---------|------|------|-------|
| `smsp__thread_inst_executed_per_inst_executed.ratio` | ratio | **平均每条指令执行的线程数** | **= 32 (无 divergence)** |
| `smsp__thread_inst_executed.avg.per_inst_executed` | threads/inst | 同上 | = 32 |

**Warp Divergence 判断**：
```
= 32:    ✅ 无 divergence，所有线程都执行
24-31:   ⚠️  轻微 divergence (~10% 浪费)
16-23:   ❌ 中等 divergence (~40% 浪费)
8-15:    ❌ 严重 divergence (~60% 浪费)
< 8:     ❌ 极严重 divergence (> 75% 浪费)

浪费率 = (32 - ratio) / 32 * 100
```

### ILP (Instruction Level Parallelism)

| 指标名称 | 单位 | 含义 | 理想值 |
|---------|------|------|-------|
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | % | 指令发射活跃度 | > 70% |
| `smsp__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active` | % | ALU 流水线利用率 | > 70% |

**ILP 不足的判断**：
```
如果：
  - SM Throughput > 80% (计算任务重)
  - 但 smsp__issue_active < 60% (指令发射低)

说明：
  - 指令发射受限，ILP 不足
  - 每个线程的独立操作太少

优化：
  - 每个线程处理多个数据
  - 循环展开
  - 向量化
```

---

## Occupancy 指标

### 占用率

| 指标名称 | 单位 | 含义 | 理想值 |
|---------|------|------|-------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | % | **实际占用率** | **> 50%** |
| `sm__maximum_warps_per_active_cycle_pct` | % | 理论最大占用率 | - |
| `sm__warps_active.avg.per_cycle_active` | warps | 平均活跃 warp 数 | 越高越好 |

### 资源限制

| 指标名称 | 单位 | 含义 | 查看方式 |
|---------|------|------|---------|
| Block Size | threads | 每个 block 的线程数 | Launch Stats |
| Registers per thread | registers | 每个线程的寄存器数 | Occupancy Section |
| Shared Memory per block | bytes | 每个 block 的共享内存 | Occupancy Section |
| Blocks per SM | blocks | 每个 SM 的 block 数 | Occupancy Section |

**占用率限制因素**：

在 NCU GUI 的 "Occupancy" section 查看 "Theoretical Occupancy Limiting Factor"

可能的限制：
1. **Registers**: 寄存器使用过多
2. **Shared Memory**: 共享内存使用过多
3. **Block Size**: Block 大小不合适
4. **Warps**: Warp 数量限制

---

## Warp State 指标

### Warp 停顿分析

| 指标名称 | 含义 | 期望值 | 问题诊断 |
|---------|------|--------|---------|
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | Active warps (正在执行) | 高 | - |
| `smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active` | 停顿在 barrier (__syncthreads) | < 20% | > 30% 同步太频繁 |
| `smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active` | 停顿在长延迟操作 (内存) | - | > 50% 内存延迟高 |
| `smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active` | 停顿在短延迟操作 (计算) | < 20% | > 30% 计算依赖 |
| `smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active` | 未被选中发射 | - | > 40% Occupancy 不足 |
| `smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_active` | 停顿在内存屏障 | < 5% | > 10% 内存一致性开销大 |
| `smsp__warps_issue_stalled_drain.avg.pct_of_peak_sustained_active` | 停顿在排空流水线 | < 10% | > 20% 频繁的控制流变化 |

**Warp State 分析决策树**：

```
停顿主要原因是什么？

├─ long_scoreboard > 50%
│  └─ 内存延迟是主要问题
│     → 优化内存访问
│     → 提高缓存命中率
│     → 使用 Shared Memory
│
├─ barrier > 30%
│  └─ 同步开销大
│     → 减少 __syncthreads() 调用
│     → 重组算法减少同步点
│
├─ short_scoreboard > 30%
│  └─ 计算依赖严重
│     → 提高 ILP
│     → 减少数据依赖
│
└─ not_selected > 40%
   └─ Occupancy 不足
      → 增加 block 数量
      → 调整资源使用
```

---

## 指标组合判断

### 快速诊断决策树

```
第一步：查看 Speed of Light
│
├─ SM > 80%, Memory < 60%
│  └─ Compute-bound
│     → 查看 Compute 指标
│     → 检查 Warp Divergence
│     → 检查 ILP
│     → 考虑 Tensor Cores
│
├─ SM < 60%, Memory > 80%
│  └─ Memory-bound
│     → 查看 Memory 指标
│     → 检查 DRAM 带宽利用率
│     → 检查 Cache 命中率
│     → 检查 Coalesced Access
│     → 检查 Bank Conflicts
│
├─ SM > 70%, Memory > 70%
│  └─ Well-balanced
│     → 接近硬件极限
│     → 从两个方向都优化
│
└─ SM < 50%, Memory < 50%
   └─ Under-utilized
      → 查看 Occupancy
      → 查看 Warp State
      → 检查 Launch Configuration
```

### Memory-bound 详细诊断

```
Memory-bound 确认后：
│
├─ DRAM bandwidth < 80% 理论值
│  ├─ Coalesced ratio > 1.5
│  │  └─ 非合并访问
│  │     → 修改访问模式
│  │     → 使用向量化 (float4)
│  │
│  ├─ L2 hit rate < 50%
│  │  └─ 数据没有复用
│  │     → 算法优化
│  │     → Tiling
│  │     → 算子融合
│  │
│  └─ L1 hit rate < 70%
│     └─ L1 cache 不友好
│        → 提高数据局部性
│        → 使用 Shared Memory
│
├─ DRAM bandwidth > 90% 理论值
│  └─ 已达到硬件极限
│     → 减少内存访问次数
│     → 算子融合
│     → 使用 Shared Memory 缓存
│
└─ Bank conflicts > 10%
   └─ Shared Memory 访问冲突
      → 添加 padding
      → 重组访问模式
```

### Compute-bound 详细诊断

```
Compute-bound 确认后：
│
├─ Warp divergence > 20%
│  └─ 分支导致浪费
│     → 重组数据
│     → 减少条件分支
│     → 使用 warp 原语
│
├─ ILP 低 (issue_active < 60%)
│  └─ 指令级并行不足
│     → 每个线程处理多个数据
│     → 循环展开
│     → 向量化
│
├─ FP32 利用率 > 80%, 但可用 Tensor Cores
│  └─ 未使用 Tensor Cores
│     → 使用 WMMA API
│     → 使用 cuBLAS/cuDNN
│
└─ 超越函数多 (exp, log, sin, cos)
   └─ 使用慢速数学函数
      → 使用 __expf, __logf 等
      → 编译选项 -use_fast_math
```

---

## 性能基准参考值

### A100 GPU 理论峰值

| 指标 | FP32 | FP16 (Tensor Core) | FP64 | INT8 |
|------|------|-------------------|------|------|
| **计算峰值** | 19.5 TFLOPS | 312 TFLOPS | 9.7 TFLOPS | 624 TOPS |
| **内存带宽** | 1555 GB/s | 1555 GB/s | 1555 GB/s | 1555 GB/s |

### 常见算子类型的典型指标

| 算子类型 | SM % | Memory % | L2 Hit | Coalesced | 备注 |
|---------|------|----------|--------|-----------|------|
| **Element-wise** | 10-20 | 80-95 | 10-30 | 1.0 | 典型 memory-bound |
| **Reduction** | 20-40 | 70-90 | 30-50 | 1.0-2.0 | Memory-bound |
| **MatMul (Naive)** | 30-50 | 60-80 | 40-60 | 1.0 | Balanced |
| **MatMul (Optimized)** | 70-90 | 40-60 | 70-90 | 1.0 | Compute-bound |
| **MatMul (Tensor Core)** | 85-95 | 30-50 | 80-95 | 1.0 | 高度优化 |
| **Convolution** | 60-80 | 50-70 | 60-80 | 1.0 | Balanced/Compute |
| **Softmax** | 15-25 | 75-90 | 20-40 | 1.0 | Memory-bound |
| **LayerNorm** | 20-35 | 70-85 | 30-50 | 1.0 | Memory-bound |

---

## 常用计算公式

### 带宽利用率

```python
# DRAM 实际带宽
total_bytes = dram__bytes_read.sum + dram__bytes_write.sum
duration_sec = gpu__time_duration.sum / 1e9
actual_bandwidth_GBps = total_bytes / duration_sec / 1e9

# 理论带宽 (GPU 特定)
theoretical_bandwidth_GBps = {
    'A100': 1555,
    'V100': 900,
    'RTX 3090': 936,
    'RTX 4090': 1008,
}

# 带宽利用率
utilization = actual_bandwidth_GBps / theoretical_bandwidth_GBps * 100
```

### 计算吞吐量

```python
# FP32 TFLOPS
fp32_ops = (
    fadd + fmul + 2 * ffma  # FMA = 2 operations
)
fp32_tflops = fp32_ops / duration_sec / 1e12

# 与理论峰值对比
peak_tflops = 19.5  # A100
efficiency = fp32_tflops / peak_tflops * 100
```

### Arithmetic Intensity

```python
# 算术强度
flops = total_floating_point_operations
bytes_accessed = total_memory_bytes_read_write

arithmetic_intensity = flops / bytes_accessed  # FLOPs/Byte

# Roofline 判断
ridge_point = peak_tflops / peak_bandwidth_GBps

if arithmetic_intensity < ridge_point:
    print("Memory-bound")
else:
    print("Compute-bound")
```

---

## 快速参考卡片

### 5 秒快速诊断

```bash
ncu --section SpeedOfLight ./program

查看：
1. SM Throughput     > 80%? → Compute-bound
2. Memory Throughput > 80%? → Memory-bound
```

### Memory-bound 快速检查

```bash
ncu --section MemoryWorkloadAnalysis ./program

查看：
1. dram__throughput           - 带宽是否饱和
2. lts__t_sector_hit_rate     - L2 命中率 > 70%?
3. l1tex__t_sector_hit_rate   - L1 命中率 > 90%?
4. l1tex__average_t_sectors_per_request - Coalesced = 1.0?
5. l1tex__data_bank_conflicts - Bank conflicts = 0?
```

### Compute-bound 快速检查

```bash
ncu --section ComputeWorkloadAnalysis ./program

查看：
1. smsp__thread_inst_executed_per_inst_executed - Divergence = 32?
2. smsp__issue_active - ILP 是否充分?
3. Tensor Core 利用率 - 如果适用
```

### Occupancy 快速检查

```bash
ncu --section Occupancy ./program

查看：
1. Achieved Occupancy  > 50%?
2. Limiting Factor     - 是什么限制了占用率?
```

---

## 总结

### 核心指标优先级

**第一级（必看）**：
1. `sm__throughput` - 判断 compute-bound
2. `gpu__compute_memory_throughput` - 判断 memory-bound

**第二级（Memory-bound 看）**：
3. `dram__throughput` - DRAM 瓶颈
4. `lts__t_sector_hit_rate` - L2 命中率
5. `l1tex__average_t_sectors_per_request` - Coalesced access

**第二级（Compute-bound 看）**：
6. `smsp__thread_inst_executed_per_inst_executed` - Warp divergence
7. `smsp__issue_active` - ILP

**第三级（低利用率看）**：
8. `sm__warps_active` - Occupancy
9. Warp State 分布 - 停顿原因

### 指标查找技巧

```bash
# 列出所有可用指标
ncu --query-metrics

# 搜索特定指标
ncu --query-metrics | grep -i "memory"
ncu --query-metrics | grep -i "occupancy"
ncu --query-metrics | grep -i "warp"

# 查看 section 包含的指标
ncu --query-sections
ncu --list-sections
```
