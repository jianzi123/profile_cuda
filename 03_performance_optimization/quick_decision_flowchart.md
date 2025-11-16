# GPU 优化快速决策流程图

## 一页纸决策指南

```
开始：我有一个慢的 kernel
│
├─ Step 0: 先 Profile！
│  └─ ncu --section SpeedOfLight ./program
│
├─ Step 1: 判断瓶颈类型（5秒决策）
│  │
│  ├─ SM > 80%, Memory < 60%
│  │  └─ Compute-bound → 跳到 [A]
│  │
│  ├─ SM < 60%, Memory > 80%
│  │  └─ Memory-bound → 跳到 [B]
│  │
│  ├─ SM > 70%, Memory > 70%
│  │  └─ Balanced → 跳到 [C]
│  │
│  └─ SM < 50%, Memory < 50%
│     └─ Under-utilized → 跳到 [D]
│
│
├─ [A] Compute-bound 优化路径
│  │
│  │  问题1: 是矩阵乘法吗？用 FP16/TF32？
│  │  └─ YES → 使用 Tensor Cores ⭐⭐⭐ (5-20x)
│  │           nvcc -arch=sm_80 + cuBLAS
│  │           验证：ncu --metrics smsp__inst_executed_pipe_tensor
│  │
│  │  问题2: Warp divergence 严重吗？
│  │  └─ ncu --metrics smsp__thread_inst_executed_per_inst_executed
│  │     └─ < 24 → 消除 Divergence ⭐⭐ (1.5-3x)
│  │              - 重组数据使同 warp 线程走相同路径
│  │              - 减少 if-else 分支
│  │
│  │  问题3: ILP 不足吗？
│  │  └─ ncu --metrics smsp__issue_active
│  │     └─ < 60% → 提高 ILP ⭐⭐ (1.3-2x)
│  │              - 每个线程处理 4 个数据
│  │              - 独立操作写成独立变量
│  │
│  │  问题4: 有超越函数吗（exp, log, sin）？
│  │  └─ YES → 使用快速数学函数 ⭐⭐ (1.5-3x)
│  │           __expf(), __logf(), rsqrtf()
│  │           或 nvcc -use_fast_math
│  │
│  └─  问题5: 有小循环（<16次）吗？
│     └─ YES → 循环展开 ⭐ (1.1-1.3x)
│              #pragma unroll
│
│
├─ [B] Memory-bound 优化路径
│  │
│  │  问题1: 有多个（>3）连续的小 kernel 吗？
│  │  └─ nsys profile 查看
│  │     └─ YES → 算子融合 ⭐⭐⭐ (3-10x) ← 最优先！
│  │              - 合并成一个 kernel
│  │              - 消除中间内存传输
│  │              - 验证：duration 大幅下降
│  │
│  │  问题2: 同一数据被多次访问吗？
│  │  └─ ncu --metrics lts__t_sector_hit_rate.pct
│  │     └─ < 50% → 使用 Shared Memory ⭐⭐⭐ (2-5x)
│  │              - Tiling 策略
│  │              - 缓存复用数据
│  │              - 注意 bank conflicts
│  │              - 验证：L2 hit rate 提升
│  │
│  │  问题3: Coalesced access 吗？
│  │  └─ ncu --metrics l1tex__average_t_sectors_per_request
│  │     └─ > 1.5 → 修复访问模式 ⭐⭐⭐ (2-4x)
│  │              - 改为连续访问
│  │              - 使用 Shared Memory 转置
│  │              - 重组数据布局
│  │              - 目标：ratio = 1.0
│  │
│  │  问题4: 已经 coalesced 但带宽还不够？
│  │  └─ 实际带宽 < 80% 理论带宽
│  │     └─ YES → 向量化 float4 ⭐⭐ (1.5-2x)
│  │              - 前提：element-wise 操作
│  │              - 前提：数据对齐
│  │              - 一次读/写 16 bytes
│  │
│  └─  问题5: Shared Memory bank conflicts？
│     └─ ncu --metrics l1tex__data_bank_conflicts
│        └─ > 0 → 消除 Bank Conflicts ⭐⭐ (1.2-1.5x)
│                 - __shared__ float data[32][33]; // +1 padding
│                 - 重组访问模式
│
│
├─ [C] Balanced（接近极限）
│  │
│  │  两方面都接近 80%，已经很好了
│  │  但如果还要优化：
│  │
│  ├─ 1. 尝试 [B] 的算子融合（如果有多个 kernel）
│  ├─ 2. 尝试 [A] 的 Tensor Cores（如果是矩阵乘法）
│  ├─ 3. 考虑算法级优化
│  └─ 4. CUDA Graphs（如果有重复执行）
│
│
└─ [D] Under-utilized（资源未充分利用）
   │
   │  问题1: Occupancy 低吗？
   │  └─ ncu --section Occupancy
   │     └─ < 40% → 提高 Occupancy ⭐⭐ (1.2-2x)
   │              - 查看 Limiting Factor
   │              - 如果是 Registers → __launch_bounds__
   │              - 如果是 Shared Memory → 减少使用
   │              - 如果是 Block Size → 调整为 256
   │
   │  问题2: Warp 大量停顿吗？
   │  └─ ncu --section WarpStateStats
   │     │
   │     ├─ stalled_barrier > 30%
   │     │  └─ 减少 __syncthreads() 调用
   │     │
   │     ├─ stalled_long_scoreboard > 50%
   │     │  └─ 优化内存访问（回到 [B]）
   │     │
   │     ├─ stalled_short_scoreboard > 30%
   │     │  └─ 提高 ILP（见 [A] 问题3）
   │     │
   │     └─ not_selected > 40%
   │        └─ 提高 Occupancy（见问题1）
   │
   └─  问题3: Launch configuration 有问题吗？
      └─ Blocks 太少？Threads 太少？
         └─ 调整 grid/block size
```

---

## 快速诊断命令

### 30秒快速诊断

```bash
# 一行命令看瓶颈
ncu --section SpeedOfLight --page raw ./program 2>&1 | grep -E "SM|Memory"

# 看输出
SM Throughput:      15%    ← 低
Memory Throughput:  88%    ← 高

# 立即知道：Memory-bound，去 [B] 路径
```

### 5分钟深入诊断

```bash
# Memory-bound 深入
ncu --section MemoryWorkloadAnalysis \
    --metrics l1tex__average_t_sectors_per_request,lts__t_sector_hit_rate \
    ./program

# Compute-bound 深入
ncu --section ComputeWorkloadAnalysis \
    --metrics smsp__thread_inst_executed_per_inst_executed,smsp__issue_active \
    ./program
```

---

## 决策矩阵

### 根据 NCU 指标直接决策

| NCU 指标 | 阈值 | 优化方法 | 优先级 |
|---------|-----|---------|--------|
| **有多个小 kernel** (nsys) | duration<10ms each | 算子融合 | ⭐⭐⭐ |
| `smsp__inst_executed_pipe_tensor` | = 0% (且是矩阵乘法) | Tensor Cores | ⭐⭐⭐ |
| `lts__t_sector_hit_rate.pct` | < 50% | Shared Memory | ⭐⭐⭐ |
| `l1tex__average_t_sectors_per_request` | > 1.5 | 修复 Coalesced | ⭐⭐⭐ |
| `dram__throughput` (理论%) | < 80% (且已coalesced) | 向量化 float4 | ⭐⭐ |
| `smsp__thread_inst_executed_per_inst_executed` | < 24 | 消除 Divergence | ⭐⭐ |
| `smsp__issue_active` | < 60% | 提高 ILP | ⭐⭐ |
| `l1tex__data_bank_conflicts.sum` | > 0 | 消除 Bank Conflicts | ⭐⭐ |
| `sm__warps_active` | < 40% | 提高 Occupancy | ⭐⭐ |
| 小循环（<16次）且固定 | - | 循环展开 | ⭐ |

---

## 示例决策过程

### 案例：慢的 ReLU + Bias + Scale 操作

```bash
# Step 1: Profile
ncu --section SpeedOfLight ./program

输出：
SM Throughput:      12%
Memory Throughput:  89%
Duration:           15 ms

# 决策：Memory-bound → 走 [B] 路径

# Step 2: 检查是否有多个 kernel
nsys profile -o timeline ./program

发现：
relu_kernel:      5 ms
add_bias_kernel:  5 ms
scale_kernel:     5 ms

# 决策：✅ 问题1 - 有 3 个连续小 kernel
# 行动：算子融合 ⭐⭐⭐

# Step 3: 融合后再 profile
ncu --section SpeedOfLight ./program_fused

输出：
Duration: 5 ms（从 15ms 降到 5ms）
加速比：3x ✓

# Step 4: 还能继续优化吗？
ncu --section MemoryWorkloadAnalysis ./program_fused

输出：
l1tex__average_t_sectors_per_request = 1.02（coalesced ✓）
实际带宽 = 800 GB/s
理论带宽 = 1555 GB/s
利用率 = 51%

# 决策：✅ 问题4 - 已经 coalesced 但带宽不够
# 行动：向量化 float4 ⭐⭐

# Step 5: 向量化后
Duration: 3.2 ms（从 5ms 降到 3.2ms）
总加速比：15/3.2 = 4.7x ✓

# 完成！
```

---

## 常见错误决策

### ❌ 错误 1：没 profile 就优化

```
开发者：我觉得这里慢，先做循环展开
（优化后）
性能：1.02x（几乎无提升）

正确做法：
1. 先 ncu profile
2. 发现是 Memory-bound
3. 循环展开对 Memory-bound 无用
4. 应该做内存优化
```

### ❌ 错误 2：优化顺序错误

```
开发者：先做循环展开（+5%），再做算子融合（+300%）

正确做法：
1. 先做算子融合（+300%）← 影响大的先做
2. 再考虑其他优化

节省时间：
错误顺序：花 1 小时循环展开 + 2 小时融合 = 3 小时
正确顺序：2 小时融合 + 评估是否需要循环展开 = 2 小时
```

### ❌ 错误 3：盲目使用高级特性

```
开发者：听说 Tensor Cores 快，我要用！
（用在 Element-wise 操作上）
性能：无提升或更慢

正确做法：
1. 检查是否是矩阵乘法
2. 不是 → 不能用 Tensor Cores
3. 应该用 Memory 优化
```

### ❌ 错误 4：优化后不验证

```
开发者：我加了 Shared Memory，应该快了
（实际导致 bank conflicts）
性能：0.8x（更慢了！）

正确做法：
1. 优化后立即 profile
2. 检查 bank conflicts
3. 发现问题，添加 padding
4. 再次验证
```

---

## 优化时间估算

基于优化技术的实施难度：

| 优化方法 | 实施时间 | 调试时间 | 总时间 | 收益 | ROI |
|---------|---------|---------|--------|------|-----|
| 算子融合 | 1-2h | 0.5h | 2h | 3-10x | ⭐⭐⭐⭐⭐ |
| Tensor Cores (cuBLAS) | 0.5h | 0.5h | 1h | 5-20x | ⭐⭐⭐⭐⭐ |
| 向量化 float4 | 1h | 0.5h | 1.5h | 1.5-2x | ⭐⭐⭐⭐ |
| Shared Memory | 2-3h | 1-2h | 4h | 2-5x | ⭐⭐⭐⭐ |
| 修复 Coalesced | 2-4h | 1h | 4h | 2-4x | ⭐⭐⭐⭐ |
| 提高 ILP | 1-2h | 0.5h | 2h | 1.3-2x | ⭐⭐⭐ |
| 消除 Divergence | 2-4h | 1h | 4h | 1.5-3x | ⭐⭐⭐ |
| 循环展开 | 0.5h | 0.5h | 1h | 1.1-1.3x | ⭐⭐ |
| Occupancy 调优 | 1-2h | 1h | 2h | 1.2-2x | ⭐⭐⭐ |

**推荐优先级（基于 ROI）**：
1. 算子融合（如适用）
2. Tensor Cores（如适用）
3. Shared Memory（如有数据复用）
4. 向量化 float4
5. 其他优化

---

## 总结：三步优化法

```
第一步（必做）：识别瓶颈
├─ ncu --section SpeedOfLight
└─ 5 秒判断 Compute/Memory bound

第二步（重点）：大的优化
├─ Memory-bound
│  ├─ 1. 算子融合（如有多个 kernel）
│  ├─ 2. Shared Memory（如有数据复用）
│  └─ 3. 修复 Coalesced（如访问跨步）
│
└─ Compute-bound
   ├─ 1. Tensor Cores（如是矩阵乘法）
   ├─ 2. 消除 Divergence（如有分支）
   └─ 3. 提高 ILP（如发射率低）

第三步（锦上添花）：小的优化
├─ 向量化
├─ 循环展开
├─ Occupancy 调优
└─ Bank Conflicts 消除

验证：每步优化后都 profile！
```

**记住**：
- ✅ Profile-driven optimization（基于数据决策）
- ✅ 优先做收益大的
- ✅ 每步都验证
- ❌ 不要猜测
- ❌ 不要过早优化
