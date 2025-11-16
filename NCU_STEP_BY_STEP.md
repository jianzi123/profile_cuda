# NCU 一步一步使用指南

> 从运行 NCU 到得出优化结论的完整流程

## 目录
1. [准备工作](#一准备工作)
2. [运行 NCU 采样](#二运行-ncu-采样)
3. [打开 NCU-UI 查看报告](#三打开-ncu-ui-查看报告)
4. [关键页面详解](#四关键页面详解)
5. [完整分析案例](#五完整分析案例)
6. [常见问题](#六常见问题)

---

## 一、准备工作

### 1.1 确认 NCU 已安装

```bash
# 检查 NCU 版本
ncu --version

# 预期输出:
# NVIDIA (R) Nsight Compute Command Line Profiler
# Copyright (c) 2018-2023 NVIDIA Corporation
# Version 2023.1.0 (build xxxxx)
```

如果没有安装：
- 方式 1: CUDA Toolkit 自带 (推荐路径: `/usr/local/cuda/bin/ncu`)
- 方式 2: 单独下载 Nsight Compute

### 1.2 确认代码可运行

```bash
# 确保你的 kernel 可以正常运行
./my_kernel 1024

# 预期看到正常输出，无 CUDA 错误
```

### 1.3 权限设置 (可能需要)

某些系统上 NCU 需要 sudo 或特殊权限：

```bash
# 如果遇到权限错误，设置允许非 root 用户 profiling
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | \
  sudo tee /etc/modprobe.d/nvidia-profiling.conf

# 重启系统使生效
sudo reboot
```

---

## 二、运行 NCU 采样

### 2.1 基础采样命令

```bash
# 最简单的命令 - 采集完整指标集
ncu --set full --export my_kernel_report ./my_kernel

# 参数说明:
# --set full         : 采集完整指标集 (推荐，包含所有重要指标)
# --export FILENAME  : 保存报告到 .ncu-rep 文件
# ./my_kernel        : 要分析的程序
```

**注意**:
- 不要加 `--force-overwrite`，避免误删之前的报告
- 程序会运行得很慢 (normal，NCU 在采集数据)

### 2.2 传递参数给程序

```bash
# 如果你的程序需要参数
ncu --set full --export gemm_1024 ./gemm 1024 1024 1024
#                                         ^^^^^^^^^^^^^^^^
#                                         程序的参数
```

### 2.3 只采集关键指标 (快速模式)

如果 `--set full` 太慢 (大型 kernel)，可以只采集关键指标：

```bash
ncu --metrics \
  gpu__time_duration.avg,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__compute_memory.avg.pct_of_peak_sustained_elapsed,\
  l1tex__average_t_sectors_per_request,\
  dram__bytes.sum \
  --export my_kernel_quick \
  ./my_kernel
```

### 2.4 采样输出示例

```bash
$ ncu --set full --export vector_add_v0 ./vector_add_v0

==PROF== Connected to process 12345
==PROF== Profiling "vector_add_naive" - 1: 0%....50%....100% - 18 passes
==PROF== Disconnected from process 12345
==PROF== Report saved to: vector_add_v0.ncu-rep
```

**关键信息**:
- `18 passes`: NCU 运行了 18 遍来采集不同指标 (正常)
- `.ncu-rep`: 生成的报告文件

---

## 三、打开 NCU-UI 查看报告

### 3.1 启动 NCU-UI

**方式 1: 命令行启动**
```bash
ncu-ui vector_add_v0.ncu-rep
```

**方式 2: GUI 启动**
- Linux: 在应用菜单找到 "NVIDIA Nsight Compute"
- Windows: 开始菜单 → NVIDIA → Nsight Compute

然后: File → Open → 选择 `.ncu-rep` 文件

### 3.2 NCU-UI 界面布局

```
┌─────────────────────────────────────────────────────────┐
│ File  View  Tools  Help                      [工具栏]   │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  页面列表    │          主内容区域                       │
│              │                                          │
│ • Details    │  ← 在这里查看各种指标和图表              │
│ • Summary    │                                          │
│ • Speed of   │                                          │
│   Light      │                                          │
│ • Memory     │                                          │
│   Workload   │                                          │
│ • ...        │                                          │
│              │                                          │
└──────────────┴──────────────────────────────────────────┘
```

**左侧页面列表**: 点击切换不同分析页面
**主内容区域**: 显示当前页面的图表和数据

---

## 四、关键页面详解

### 4.1 页面 "Details" - 基本信息

**位置**: 左侧页面列表 → Details

**内容**:
```
Kernel Name: vector_add_naive
Duration: 12.450 ms
Grid Size: (4096, 1, 1)
Block Size: (256, 1, 1)
Registers Per Thread: 16
Shared Memory: 0 bytes
...
```

**看什么**:
- `Duration`: 执行时间 (这是你要优化的目标)
- `Grid/Block Size`: 确认 launch configuration 正确

---

### 4.2 页面 "GPU Speed Of Light Throughput" - **最重要**

**位置**: 左侧页面列表 → GPU Speed Of Light Throughput

#### 4.2.1 页面布局

```
GPU Speed Of Light Throughput
─────────────────────────────────────────

┌─────────────────────────────────────┐
│  SM Throughput        █░░░░ 12.3%   │ 🔴 低
│  Memory Throughput    █████████ 89.2% │ 🔴 高
└─────────────────────────────────────┘

Performance Limiter: Memory Throughput (89.2%)
```

#### 4.2.2 如何看这个图表

**进度条颜色**:
- 🟢 绿色 (>80%): 利用率高
- 🟡 黄色 (60-80%): 中等
- 🟠 橙色 (40-60%): 较低
- 🔴 红色 (<40%): 很低

**上面的例子解读**:
```
SM Throughput: 12.3% 🔴
  → 计算单元只用了 12%，大量闲置

Memory Throughput: 89.2% 🔴
  → 显存带宽快用满了 (瓶颈!)

结论: Memory-bound (显存瓶颈)
```

#### 4.2.3 四象限判断法

| SM Throughput | Memory Throughput | 诊断 |
|---------------|-------------------|------|
| 低 (<40%) | **高 (>60%)** | **Memory-bound** ← 最常见 |
| **高 (>60%)** | 低 (<40%) | **Compute-bound** |
| 低 | 低 | Launch-bound (并行度不足) |
| 高 | 高 | Balanced (已优化) |

**实战例子**:
```
Example 1:
SM: 8%, Memory: 92% → Memory-bound，优化内存访问

Example 2:
SM: 75%, Memory: 25% → Compute-bound，优化计算 (Tensor Core)

Example 3:
SM: 15%, Memory: 18% → Launch-bound，增加 blocks/threads

Example 4:
SM: 85%, Memory: 82% → 已优化，转向系统级优化
```

---

### 4.3 页面 "Memory Workload Analysis" - 内存诊断

**位置**: 左侧页面列表 → Memory Workload Analysis

#### 4.3.1 这个页面有什么

```
Memory Workload Analysis
─────────────────────────────────────────

┌─ L1/TEX Cache ────────────────────────┐
│                                       │
│ [Table with many rows]                │
│                                       │
│ Row: l1tex__average_t_sectors_per_request │
│      Value: 32.00                     │ ← 重点看这个!
│                                       │
│ Row: l1tex__t_bytes_per_sector...     │
│      Value: 4.00 byte                 │
│                                       │
└───────────────────────────────────────┘

┌─ L2 Cache ────────────────────────────┐
│ lts__t_sector_hit_rate: 2.3%          │ ← L2 命中率
└───────────────────────────────────────┘

┌─ Device Memory (DRAM) ────────────────┐
│ dram__bytes.sum: 8,388,608,000 bytes  │ ← 总显存流量
│                  (8.0 GB)             │
└───────────────────────────────────────┘
```

#### 4.3.2 关键指标详解

##### 指标 1: `l1tex__average_t_sectors_per_request`

**在哪找**:
1. 打开 "Memory Workload Analysis" 页面
2. 找到 "L1/TEX Cache" 区域
3. 滚动表格，找到行 `l1tex__average_t_sectors_per_request`
4. 查看右侧的数值

**这个指标是什么**:
- 每次内存请求平均访问多少个 sector (32-byte 块)
- **理想值**: 1.0 (完美合并)
- **异常值**: > 1.5 (访问未合并，浪费带宽)

**计算带宽浪费**:
```
sectors_per_request = 32.0

带宽浪费 = (32.0 - 1.0) / 32.0 = 96.9%
         → 96.9% 的带宽被浪费了!
```

**原因**: warp 内 32 个线程访问了 32 个不同的 cache line

**解决**: 修复 Memory Coalescing

##### 指标 2: `dram__bytes.sum`

**在哪找**:
1. 同一页面
2. 找到 "Device Memory" 或 "DRAM" 区域
3. 查看 `dram__bytes.sum`

**这个指标是什么**:
- 总显存流量 (读+写)

**如何判断异常**:
```
理论流量计算 (以 GEMM 1024×1024×1024 为例):
  A: 1024×1024×4 bytes = 4 MB
  B: 1024×1024×4 bytes = 4 MB
  C: 1024×1024×4 bytes = 4 MB (写)
  理论总计: 12 MB

NCU 实际显示: dram__bytes.sum = 8 GB

问题: 8000 MB / 12 MB = 667 倍重复读取!
```

**原因**: 没有使用 Shared Memory 缓存，每次都从显存读

**解决**: Shared Memory Tiling

##### 指标 3: `lts__t_sector_hit_rate`

**在哪找**:
1. "L2 Cache" 区域
2. 查看 `lts__t_sector_hit_rate`

**这个指标是什么**:
- L2 Cache 命中率

**正常范围**:
- Streaming 访问 (如 Vector Add): 5-10% (正常，数据只读一次)
- 有局部性 (如小矩阵): 40-60%
- Tile 重用: 60-80%

**不需要刻意优化 L2**: 如果访问模式合理，L2 会自动工作

---

### 4.4 页面 "Warp State Statistics" - Stall 分析

**位置**: 左侧页面列表 → Scheduler Statistics → Warp State Statistics

#### 4.4.1 页面内容

```
Warp State Statistics
─────────────────────────────────────────

Average Warp States (%)
┌───────────────────────────┐
│ Active:           15.2%   │ 🔴 太低
│ Stalled:          84.8%   │
│                           │
│ Stall Breakdown:          │
│ ├─ Long Scoreboard: 68.5% │ ← 重点!
│ ├─ Short Scoreboard: 8.2% │
│ ├─ Barrier:          2.1% │
│ ├─ Not Selected:     5.3% │
│ └─ Other:            0.7% │
└───────────────────────────┘
```

#### 4.4.2 指标含义

**Active (活跃)**:
- Warp 正在执行指令
- **期望**: > 40% (越高越好)
- **低于 20%**: 说明大量时间在等待

**Stalled (停顿)**: Warp 在等待，细分为：

##### Long Scoreboard (长延迟 stall)
- **原因**: 等待显存访问 (DRAM latency ~400 cycles)
- **正常值**: Memory-bound kernel 60-80% 是正常的
- **异常值**: > 80% → 显存访问太慢

**如何降低**:
- 修复 Memory Coalescing
- 使用 Shared Memory
- 增加 Occupancy (隐藏延迟)

##### Barrier (同步 stall)
- **原因**: 等待 `__syncthreads()`
- **正常值**: < 10%
- **异常值**: > 20% → 同步太频繁或 warp divergence

**如何降低**:
- 减少 `__syncthreads()` 次数
- 检查是否误用 Shared Memory (如 Vector Add v3)

##### Short Scoreboard (短延迟 stall)
- **原因**: 等待寄存器或 Shared Memory 访问
- **正常值**: < 15%

##### Not Selected (未被调度)
- **原因**: 其他 warp 在运行
- **正常值**: 任意 (调度器自动处理)

#### 4.4.3 实际案例解读

**Case 1: Vector Add v0 (Memory-bound)**
```
Active: 15.2%
Long Scoreboard: 68.5% ← 大部分时间在等显存
Barrier: 2.1%

诊断: 显存访问慢 (sectors_per_request=32)
方案: 修复 Coalescing
```

**Case 2: Vector Add v3 (错误使用 Shared Memory)**
```
Active: 22.1%
Long Scoreboard: 52.4%
Barrier: 28.7% ← 异常高! (v2 只有 5%)

诊断: __syncthreads() 引入过多同步开销
方案: 移除 Shared Memory (无数据重用)
```

---

### 4.5 页面 "Occupancy" - 占用率分析

**位置**: 左侧页面列表 → Occupancy

#### 4.5.1 页面内容

```
Occupancy
─────────────────────────────────────────

Theoretical Occupancy:   100%
Achieved Occupancy:       42.3%

Limiting Factors:
┌────────────────────────────────┐
│ ✓ Block Size: 256              │
│ ✓ Registers: 24 per thread     │
│ ⚠ Shared Memory: 48 KB         │ ← 限制因素!
│   (Max 96 KB per SM)           │
│   → Can run 2 blocks per SM    │
│   → Theoretical: 100%          │
│   → But achieved only 42%      │
└────────────────────────────────┘
```

#### 4.5.2 Occupancy 是什么

**定义**: SM 上活跃 warp 占理论最大 warp 的比例

**A100 SM 规格**:
- 每个 SM 最多 64 个 warp (2048 threads)
- 如果只有 27 个 warp 活跃 → Occupancy = 27/64 = 42%

#### 4.5.3 Occupancy 越高越好吗？

**❌ 常见误区**: "Occupancy 必须 100%"

**✅ 正确理解**:
```
Occupancy 的作用: 隐藏延迟

Memory-bound kernel:
  Occupancy 50% 就够了 (足够隐藏 DRAM latency)

Compute-bound kernel:
  需要更高 Occupancy (70-90%)

已经 90% 带宽效率的 kernel:
  提升 Occupancy 没用 (瓶颈不在这)
```

**实际案例**:
```
Kernel A: Occupancy 45%, Bandwidth 92%
  → 不需要优化 Occupancy

Kernel B: Occupancy 45%, SM 85%, Memory 30%
  → 需要提升 Occupancy (计算受限)
```

#### 4.5.4 如何提升 Occupancy

**看 Limiting Factors**:

```
If Limiting Factor = Registers:
  → 减少寄存器使用
     nvcc -maxrregcount=64

If Limiting Factor = Shared Memory:
  → 减少 Shared Memory 使用
  → 或增加 block size

If Limiting Factor = Block Size:
  → 调整 block size (try 128, 256, 512)
```

---

## 五、完整分析案例

### 案例 1: Vector Add v0 → v1 优化

#### Step 1: 运行 NCU

```bash
ncu --set full --export vector_add_v0 ./vector_add_v0
```

#### Step 2: 打开 NCU-UI

```bash
ncu-ui vector_add_v0.ncu-rep
```

#### Step 3: 查看 Speed of Light

**位置**: 左侧 → GPU Speed Of Light Throughput

**看到**:
```
SM Throughput:      12.3% 🔴
Memory Throughput:  89.2% 🔴
```

**结论**: Memory-bound (显存瓶颈)

#### Step 4: 查看 Memory Workload Analysis

**位置**: 左侧 → Memory Workload Analysis

**找到指标**:
```
L1/TEX Cache 表格中:
  l1tex__average_t_sectors_per_request: 32.00 🔴

计算:
  带宽浪费 = (32 - 1) / 32 = 96.9%
```

**结论**: 非合并访问，浪费 96.9% 带宽

#### Step 5: 查看 Warp State Statistics

**位置**: 左侧 → Scheduler Statistics → Warp State Statistics

**看到**:
```
Long Scoreboard Stall: 68.5%
```

**结论**: 大量时间在等待显存 (验证了 Memory-bound)

#### Step 6: 诊断总结

```
瓶颈: Memory-bound
根因: 非合并访问 (sectors_per_request = 32)
优化: 修复 Memory Coalescing
预期: 8-10x 提升
```

#### Step 7: 代码修复

```c
// 修改前 (v0)
int stride = 32;
c[tid * stride] = a[tid * stride] + b[tid * stride];

// 修改后 (v1)
c[tid] = a[tid] + b[tid];
```

#### Step 8: 验证优化

```bash
# 重新采样
ncu --set full --export vector_add_v1 ./vector_add_v1

# 打开对比
ncu-ui vector_add_v0.ncu-rep vector_add_v1.ncu-rep
```

**对比结果**:
```
                          v0          v1
Duration:                12.5 ms     1.5 ms     (8.3x 提升!) ✅
sectors_per_request:     32.0        1.0        (修复!) ✅
Memory Throughput:       89%         92%
SM Throughput:           12%         28%        (提升)
```

**结论**: 优化成功! 性能提升 8.3x

---

### 案例 2: GEMM Shared Memory 优化

#### Step 1: v0 NCU 诊断

```bash
ncu --set full --export gemm_v0 ./gemm_v0 1024 1024 1024
ncu-ui gemm_v0.ncu-rep
```

#### Step 2: Speed of Light 分析

```
SM Throughput:      8.1% 🔴
Memory Throughput:  91.8% 🔴
```

**结论**: Memory-bound

#### Step 3: Memory Workload 深入分析

```
L1/TEX Cache:
  sectors_per_request: 1.2 ✅ (访问基本合并)

Device Memory:
  dram__bytes.sum: 8,388,608,000 bytes (8 GB) 🔴

理论计算:
  A: 1024×1024×4 = 4 MB
  B: 1024×1024×4 = 4 MB
  C: 1024×1024×4 = 4 MB
  理论: 12 MB

实际 / 理论 = 8000 / 12 = 667x 重复读取!
```

**结论**: 虽然访问合并了，但重复读取严重

#### Step 4: 分析原因

查看代码:
```c
// v0 naive
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
}

问题:
  - A[row * K + k] 被读 N = 1024 次 (每个 column)
  - B[k * N + col] 被读 M = 1024 次 (每个 row)
  → 每个元素重复读取 1024 次!
```

#### Step 5: 优化方案

**方案**: Shared Memory Tiling - 缓存 tile 实现重用

#### Step 6: 实施优化 (v2)

```c
__shared__ float As[32][33];  // +1 padding
__shared__ float Bs[32][33];

for (int t = 0; t < K/32; t++) {
    // 加载 tile (协作加载)
    As[ty][tx] = A[...];
    Bs[ty][tx] = B[...];
    __syncthreads();

    // 计算 (重用 32 次)
    for (int k = 0; k < 32; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
}
```

#### Step 7: 验证

```bash
ncu --set full --export gemm_v2 ./gemm_v2 1024 1024 1024
```

**对比**:
```
                     v0              v2
Duration:           14.3 ms          0.86 ms    (16.6x!) ✅
dram__bytes:        8.0 GB           8.0 MB     (1000x 减少!) ✅
SM Throughput:      8%               28%        (提升)
Performance:        150 GFLOPS       2500 GFLOPS
```

**结论**: Shared Memory 成功减少重复读取，性能提升 16x

---

## 六、常见问题

### Q1: NCU 运行太慢怎么办？

**A**: 使用快速模式，只采集关键指标

```bash
# 只采集 5 个最重要指标
ncu --metrics \
  gpu__time_duration.avg,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__compute_memory.avg.pct_of_peak_sustained_elapsed,\
  l1tex__average_t_sectors_per_request,\
  dram__bytes.sum \
  --export quick_report \
  ./my_kernel
```

或减小问题规模：
```bash
# 用小数据测试
./my_kernel 256  # instead of 1024
```

### Q2: 找不到 `l1tex__average_t_sectors_per_request` 指标

**A**:
1. 确保使用 `--set full`
2. 在 Memory Workload Analysis 页面
3. 滚动 "L1/TEX Cache" 表格
4. 如果还是没有，尝试更新 NCU 版本

### Q3: 多个 kernel 怎么看？

**A**: NCU-UI 左上角有 kernel 选择下拉框

```
┌─────────────────────────┐
│ Kernel: [vector_add_naive ▼] │ ← 点击选择
├─────────────────────────┤
│ Call 1 of 1             │
└─────────────────────────┘
```

### Q4: 如何对比两个版本？

**A**:
```bash
# 方式 1: 同时打开两个报告
ncu-ui v0.ncu-rep v1.ncu-rep

# 方式 2: 使用 baseline comparison
ncu-ui v1.ncu-rep --baseline v0.ncu-rep
```

在 UI 中会显示差异：
```
Duration: 1.5 ms (↓ 8.3x from baseline) 🟢
```

### Q5: sectors_per_request 已经是 1.0，还是慢怎么办？

**A**:
1. 检查 `dram__bytes.sum` 是否远大于理论值 → Shared Memory Tiling
2. 检查 SM Throughput 是否很低 → 可能需要 Tensor Core
3. 检查 Occupancy 是否太低 → 调整 launch config

### Q6: 命令行模式能看指标吗？

**A**: 可以，但不如 GUI 直观

```bash
# 输出关键指标到终端
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__compute_memory.avg.pct_of_peak_sustained_elapsed,\
  l1tex__average_t_sectors_per_request \
  ./my_kernel

# 输出示例:
# sm__throughput.avg.pct_of_peak_sustained_elapsed      12.3 %
# gpu__compute_memory.avg.pct_of_peak_sustained_elapsed 89.2 %
# l1tex__average_t_sectors_per_request                  32.00
```

### Q7: 权限错误 "ERR_NVGPUCTRPERM"

**A**:
```bash
# 临时解决 (需要 sudo)
sudo ncu ...

# 永久解决
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | \
  sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo reboot
```

---

## 七、快速参考卡

### 7.1 必看的 3 个页面

| 页面 | 看什么 | 得到什么 |
|------|--------|----------|
| **Speed of Light** | SM% vs Memory% | 瓶颈类型 (5 秒) |
| **Memory Workload** | sectors_per_request | 是否需要 Coalescing (10 秒) |
| **Warp State** | Long Scoreboard % | 验证瓶颈 (5 秒) |

**总计**: 20 秒诊断

### 7.2 关键指标阈值

| 指标 | 正常 | 需要优化 |
|------|------|----------|
| sectors_per_request | < 1.5 | > 1.5 → Coalescing |
| Memory Throughput | < 60% | > 80% → Memory-bound |
| SM Throughput | 任意 | < 20% 且 Memory > 80% |
| dram__bytes | 接近理论 | >> 理论 → Tiling |
| Barrier Stall | < 10% | > 20% → 减少 sync |
| Occupancy | > 40% | < 30% → 调整 config |

### 7.3 典型优化路径

```
NCU 诊断
  ↓
sectors > 1.5?
  ↓ Yes
修复 Coalescing → 重新 NCU
  ↓ sectors = 1.0
dram__bytes >> 理论?
  ↓ Yes
Shared Memory Tiling → 重新 NCU
  ↓ dram__bytes ≈ 理论
效率 > 80%?
  ↓ Yes
停止 kernel 优化
转向: Tensor Core / cuBLAS / 系统级优化
```

---

## 八、总结

### 核心流程 (3 步，20 秒)

```bash
1. 运行: ncu --set full --export report ./kernel
2. 打开: ncu-ui report.ncu-rep
3. 诊断:
   - Speed of Light → Memory-bound?
   - sectors_per_request → > 1.5?
   - 得出结论 → 优化方向
```

### 记住

- **不要陷入细节**: 只看核心 2-3 个指标
- **快速迭代**: 诊断 → 优化 → 验证 (每轮 30 分钟)
- **知道停手**: 效率 > 80% 时转向其他优化

---

**下一步**: 查看 `GPU_OPTIMIZATION_ESSENCE.md` 了解具体优化技术
