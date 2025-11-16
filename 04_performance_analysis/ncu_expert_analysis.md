# NCU 专家级分析指南

> **核心思想**：NCU 不是指标的集合，而是一个推理系统。每个指标都是线索，专家的工作是通过这些线索重建性能故事。

---

## 目录

1. [NCU 分析的思维模型](#ncu-分析的思维模型)
2. [分层分析方法论](#分层分析方法论)
3. [核心指标深度解析](#核心指标深度解析)
4. [指标交叉验证技巧](#指标交叉验证技巧)
5. [实战案例：完整分析流程](#实战案例完整分析流程)
6. [常见误区与陷阱](#常见误区与陷阱)
7. [高级技巧](#高级技巧)

---

## NCU 分析的思维模型

### 新手 vs 专家的差异

```python
# ❌ 新手思维
"我看到 SM Utilization = 60%，所以还有优化空间"
→ 盲目提高 occupancy
→ 性能没提升，甚至下降

# ✅ 专家思维
"SM Utilization = 60%，但要先看：
 1. Warp Stall Reasons - 为什么 40% 的 warp 在停顿？
 2. Issue Efficiency - 是真的在计算，还是在等待？
 3. Memory Throughput - 是否被内存限制？
 4. Achieved Occupancy vs Theoretical - 差距在哪？
 → 综合分析后，发现是 long scoreboard stall (内存等待)
 → 真正的问题是内存访问，不是 occupancy
 → 优化方向：改善内存访问模式"
```

### NCU 的三层分析金字塔

```
                    ┌─────────────────┐
                    │   Level 1       │
                    │  Speed of Light │  ← 5 分钟：瓶颈类型
                    │  (宏观诊断)      │
                    └─────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │         Level 2                 │
          │   Section Analysis              │  ← 20 分钟：瓶颈定位
          │   (Memory/Compute/Scheduler)    │
          └─────────────────────────────────┘
                       │
    ┌──────────────────┴──────────────────┐
    │          Level 3                    │
    │    Detailed Metrics                 │  ← 1 小时：根因分析
    │    (单个 counter 深度分析)           │
    └─────────────────────────────────────┘
```

**关键原则**：
1. **自顶向下**：先看 Speed of Light，确定大方向
2. **交叉验证**：单个指标可能误导，多个指标交叉确认
3. **量化推理**：用数学验证假设

---

## 分层分析方法论

### Level 1: Speed of Light Analysis (5 分钟)

**目标**：快速判断瓶颈类型

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./program
```

**核心指标**：

#### 1. SM Throughput (计算吞吐量)

```
定义：SM (Streaming Multiprocessor) 的活跃程度
公式：(Active Cycles / Total Cycles) × 100%

解读：
- > 80%: Compute-bound (计算瓶颈)
- 60-80%: Balanced (均衡)
- 40-60%: 可能被其他因素限制
- < 40%: 明显的非计算瓶颈
```

**深度理解**：

```
SM Throughput 不等于"计算利用率"！

SM Active 包括：
- 真正在执行指令的时间
- 等待 scoreboard 的时间 (指令依赖)
- 等待 barrier 的时间

所以：SM Throughput = 80% 不代表 80% 时间在计算
      需要进一步看 Issue Efficiency 和 Warp Stall
```

#### 2. Memory Throughput (内存吞吐量)

```
定义：内存子系统的活跃程度
公式：(Memory Requests / Peak Memory Bandwidth) × 100%

解读：
- > 80%: Memory-bound (内存瓶颈)
- 60-80%: 内存压力较大
- < 60%: 不是内存瓶颈

细分：
- DRAM Throughput: HBM 带宽利用率
- L2 Throughput: L2 cache 带宽利用率
- L1/TEX Throughput: L1/Texture cache 带宽利用率
```

**专家洞察**：

```python
# 案例：Memory Throughput = 90%, SM Throughput = 15%

# 新手结论：
"明显是 memory-bound，优化内存访问"

# 专家分析：
"等等，先看细节：
 - DRAM Throughput = 90%
 - L2 Hit Rate = ?
 - L1 Hit Rate = ?

 如果 L2 Hit Rate = 95%，说明数据在 L2 缓存中
 但 DRAM Throughput 还是 90%，这不合理！

 进一步看：
 - lts__t_sectors_srcunit_tex_op_read (L2 从哪里读)

 发现大量数据是 L2 写回 DRAM！
 → 真正问题：中间结果写回内存
 → 优化方向：算子融合，避免写回"
```

#### 3. 四象限分析法

```
        SM Throughput
             ↑
         80% │
             │  Balanced      Compute-bound
             │  (接近极限)     (优化计算)
             │
             │
─────────────┼─────────────────────────►
         80% │              Memory Throughput
             │
             │ Under-utilized  Memory-bound
             │ (资源浪费)       (优化内存)
             │
```

**决策表**：

| SM% | Memory% | 类型 | 第一步优化方向 |
|-----|---------|------|--------------|
| >80 | <60 | Compute-bound | 查看 Compute Workload Analysis |
| <60 | >80 | Memory-bound | 查看 Memory Workload Analysis |
| >70 | >70 | Balanced | 两方面都接近极限，优先算法级优化 |
| <50 | <50 | Under-utilized | 查看 Scheduler/Launch Statistics |

---

### Level 2: Section Analysis (20 分钟)

#### A. Memory-bound 深入分析

**Section**: `MemoryWorkloadAnalysis`

```bash
ncu --section MemoryWorkloadAnalysis \
    --metrics lts__t_sector_hit_rate,l1tex__t_sector_hit_rate,\
              l1tex__average_t_sectors_per_request,\
              l1tex__data_bank_conflicts_pipe_lsu \
    ./program
```

##### 指标 1: L2 Cache Hit Rate (`lts__t_sector_hit_rate`)

```
定义：L2 cache 命中率
公式：L2 Hits / (L2 Hits + L2 Misses) × 100%

硬件背景：
- A100: L2 = 40 MB, 带宽 ~5 TB/s
- 访问延迟：L2 hit ~200 cycles, L2 miss ~400 cycles (HBM)

解读：
- > 90%: 极好，数据复用很高
- 70-90%: 良好
- 50-70%: 一般，有优化空间
- < 50%: 差，几乎没有数据复用
```

**深度分析**：

```python
# 场景 1: L2 Hit Rate = 95%，但性能仍慢

# 可能原因：
1. L2 带宽饱和
   → 检查 lts__t_sectors (L2 吞吐量)
   → 如果接近 5 TB/s，说明 L2 带宽是瓶颈

2. 访问模式不佳，虽然命中但效率低
   → 检查 lts__t_sectors_srcunit_tex_op_read_hit_rate
   → 查看是否有大量 sector 浪费

3. L2 → L1 传输瓶颈
   → 检查 l1tex__lsu_writeback_active
```

```python
# 场景 2: L2 Hit Rate = 20%，如何优化？

# 推理过程：
L2 Size = 40 MB
Working Set Size = ?

# 计算 Working Set：
ncu --metrics dram__bytes_read.sum ./program
# 假设输出：dram__bytes_read = 2 GB

Working Set >> L2 Size
→ L2 无法缓存所有数据

# 优化方向：
1. Tiling: 将数据分块，每块 < L2 Size
2. 算子融合: 减少数据多次读取
3. Shared Memory: 用 SRAM 替代 L2
```

##### 指标 2: L1/TEX Cache Hit Rate (`l1tex__t_sector_hit_rate`)

```
定义：L1/Texture cache 命中率
硬件：A100 每个 SM 有 192 KB L1/Shared Memory

解读：
- > 95%: 极好
- 90-95%: 良好
- 80-90%: 一般
- < 80%: 差，大量 L1 miss
```

**L1 vs L2 的关系**：

```
场景 1: L1 Hit = 95%, L2 Hit = 30%
→ 解释：数据在当前 SM 的 L1 中复用很好
        但跨 kernel/跨 SM 没有复用
→ 优化：局部访问已优化，全局可以算子融合

场景 2: L1 Hit = 50%, L2 Hit = 90%
→ 解释：数据在 L2 复用好，但 L1 内没有复用
        可能是 warp 之间访问模式冲突
→ 优化：调整访问模式，增加 warp 内复用

场景 3: L1 Hit = 30%, L2 Hit = 25%
→ 解释：完全没有数据复用，流式访问
→ 优化：这是 element-wise 操作的特征
        优化方向：coalescing + 向量化
```

##### 指标 3: Memory Access Coalescing (`l1tex__average_t_sectors_per_request`)

```
定义：每次内存请求平均访问的 sector 数量
理想值：1.0 (完美 coalesced)

硬件背景：
- 一个 sector = 32 bytes
- 一个 warp (32 threads) 发出一个请求
- 理想情况：32 threads 访问连续 32×4 = 128 bytes = 4 sectors
- 但 GPU 以 128-byte cache line 对齐

公式：
sectors_per_request = 实际访问的 sectors / 请求数量
```

**深度解读**：

```c
// 案例 1: 完美 coalesced (ratio = 1.0)
__global__ void perfect_coalesce(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];  // 连续访问
    }
}

// 线程 0: data[0]
// 线程 1: data[1]
// ...
// 线程 31: data[31]
// → 32 个 float = 128 bytes = 4 sectors → 1 个请求
// → sectors_per_request = 4 / 1 = 4.0... 等等，为什么不是 1.0？

// 正确理解：
// NCU 的定义是"每个请求访问多少个 cache line (128-byte aligned)"
// 如果 32 threads 访问的数据完全在 1 个 cache line 内 → 1.0
// 如果跨越 2 个 cache line → 2.0
// 如果每个 thread 访问不连续 → 可能 32.0！
```

```c
// 案例 2: 跨步访问 (ratio = 32.0)
__global__ void strided_access(float* data, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx * stride];  // stride = 64
    }
}

// 线程 0: data[0]
// 线程 1: data[64]    (256 bytes 之后)
// ...
// → 每个 thread 访问一个不同的 cache line
// → 32 个请求，访问 32 个 cache line
// → sectors_per_request = 32

// 性能影响：
// 理想：1 个请求，128 bytes 传输
// 实际：32 个请求，32 × 128 = 4096 bytes 传输
// 浪费：(4096 - 128) / 4096 = 96.875% 带宽浪费！
```

**量化分析**：

```python
# 给定 sectors_per_request = r
带宽效率 = 1 / r

# 例子：
r = 1.0 → 效率 = 100%
r = 2.0 → 效率 = 50%
r = 4.0 → 效率 = 25%
r = 32.0 → 效率 = 3.125%

# 实际带宽：
理论峰值 = 1.5 TB/s (A100)
实际带宽 = 理论峰值 × (1 / r)

# 如果测到 r = 3.2：
实际带宽 = 1.5 / 3.2 = 0.47 TB/s
损失 = 1.5 - 0.47 = 1.03 TB/s (69% 带宽浪费！)
```

##### 指标 4: Shared Memory Bank Conflicts (`l1tex__data_bank_conflicts_pipe_lsu`)

```
定义：Shared Memory bank conflict 的数量
硬件：Shared Memory 分为 32 个 bank，每个 bank 每周期服务 1 个请求

理想值：0 (无 conflict)
```

**硬件原理**：

```
Shared Memory Layout (每个 bank 宽度 4 bytes):
Bank 0: [0] [32] [64] [96] ...
Bank 1: [1] [33] [65] [97] ...
Bank 2: [2] [34] [66] [98] ...
...
Bank 31: [31] [63] [95] [127] ...

地址到 bank 的映射：
bank_id = (address / 4) % 32
```

**案例分析**：

```c
// 案例 1: 无 conflict
__shared__ float data[32];

// Warp 中 32 个 thread
int tid = threadIdx.x;
float val = data[tid];  // Thread i 访问 bank i

// 每个 thread 访问不同的 bank → 无 conflict ✓

// ─────────────────────────────────────────────

// 案例 2: 32-way conflict (最坏情况)
__shared__ float data[1024];

int tid = threadIdx.x;
float val = data[tid * 32];  // stride = 32

// Thread 0: data[0]   → bank 0
// Thread 1: data[32]  → bank 0
// Thread 2: data[64]  → bank 0
// ...
// 所有 thread 都访问 bank 0！

// 硬件行为：串行化，32 个周期完成
// 性能损失：32x

// ─────────────────────────────────────────────

// 案例 3: 2-way conflict
__shared__ float data[32][32];

int tid = threadIdx.x;
float val = data[0][tid];  // 访问第一行

// 如果行优先存储：
// Thread 0: data[0][0]   → address 0   → bank 0
// Thread 1: data[0][1]   → address 4   → bank 1
// ...
// Thread 31: data[0][31] → address 124 → bank 31
// → 无 conflict ✓

float val2 = data[tid][0];  // 访问第一列

// Thread 0: data[0][0]   → bank 0
// Thread 1: data[1][0]   → bank 8  (32 * 1 / 4 = 8)
// Thread 2: data[2][0]   → bank 16 (32 * 2 / 4 = 16)
// ...
// → 有规律的 conflict！
```

**Padding 技巧**：

```c
// ❌ 有 conflict
__shared__ float tile[32][32];  // 每行 32 × 4 = 128 bytes

// 列访问时：
// 相邻两行的第一个元素相差 32 个 float = 128 bytes
// 128 / 4 = 32 → 映射到相同的 bank

// ✅ 无 conflict (添加 padding)
__shared__ float tile[32][33];  // 每行 33 个元素

// 现在相邻两行相差 33 × 4 = 132 bytes
// 132 / 4 = 33
// 33 % 32 = 1 → 不同的 bank ✓

// 代价：浪费 32 × 4 = 128 bytes per tile (可接受)
// 收益：2-4x 加速 (Shared Memory 密集的 kernel)
```

**量化影响**：

```bash
# 测量 bank conflict
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu.sum,\
              smsp__sass_average_data_bytes_per_sector_mem_shared.pct \
    ./program

# 输出：
# bank_conflicts = 1,000,000
# shared_accesses = 10,000,000
# conflict_ratio = 10%

# 性能影响：
# 假设无 conflict 时间 = T
# 有 10% conflict (假设都是 2-way):
# 实际时间 = T × (1 + 0.1 × 1) = 1.1T
# 性能损失 = 10%

# 如果是 32-way conflict:
# 实际时间 = T × (1 + 0.1 × 31) = 4.1T
# 性能损失 = 310%！
```

---

#### B. Compute-bound 深入分析

**Section**: `ComputeWorkloadAnalysis`

```bash
ncu --section ComputeWorkloadAnalysis \
    --metrics smsp__inst_executed_pipe_tensor,\
              smsp__thread_inst_executed_per_inst_executed,\
              smsp__issue_active.pct,\
              smsp__warp_issue_stalled_long_scoreboard.pct \
    ./program
```

##### 指标 1: Tensor Core Utilization (`smsp__inst_executed_pipe_tensor`)

```
定义：Tensor Core 执行的指令数
适用：矩阵乘法、卷积等

硬件背景：
- A100: 每个 SM 有 4 个 Tensor Core
- FP16 Tensor Core: 256 FLOPS/cycle per SM
- 总计: 108 SM × 256 = 27,648 FLOPS/cycle
- @ 1.41 GHz = 312 TFLOPS
```

**深度分析**：

```python
# 场景 1: tensor_inst = 0，但是矩阵乘法 kernel

# 原因 1: 数据类型不支持
# Tensor Core 要求：FP16, BF16, TF32, INT8
# 如果用 FP32 → 不会使用 Tensor Core

# 原因 2: 矩阵大小不是 Tensor Core 友好的
# Tensor Core 最优尺寸：M, N, K 是 16 的倍数 (Ampere)
# 如果 M=17, N=17, K=17 → 效率低

# 原因 3: 没有调用 WMMA API
# 手写 GEMM 不会自动使用 Tensor Core
# 需要用 cuBLAS 或 WMMA intrinsics

# 优化方向：
1. 使用 FP16/TF32
2. Padding 到 Tensor Core 友好的尺寸
3. 使用 cuBLAS 或 cutlass
```

```python
# 场景 2: tensor_inst > 0，但利用率低

# 检查：
ncu --metrics smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active

# 如果 < 50%，可能原因：

1. Occupancy 不足
   → 增加 blocks 或 threads per block

2. 数据供应不足 (memory-bound)
   → 虽然是 Tensor Core，但被内存限制
   → 检查 Memory Throughput

3. Tensor Core 配置不优
   → 使用 auto-tuning (cutlass)
```

##### 指标 2: Warp Divergence (`smsp__thread_inst_executed_per_inst_executed`)

```
定义：平均每条指令执行多少个 thread
理想值：32 (一个 warp 所有 thread 都执行)

公式：
divergence_efficiency = actual / 32 × 100%
```

**硬件原理**：

```
SIMT 执行模型：
- 一个 warp = 32 threads
- 同一时刻所有 threads 执行相同指令 (SIMD)
- 但可以用 predication mask 关闭部分 thread

示例：
if (threadIdx.x < 16) {
    // Branch A
} else {
    // Branch B
}

执行过程：
1. Warp 发射 Branch A 指令，mask = 0xFFFF (前 16 个 thread)
2. Warp 发射 Branch B 指令，mask = 0xFFFF0000 (后 16 个 thread)
3. 总共 2 个指令周期，但只有 50% 效率
```

**案例分析**：

```c
// 案例 1: 严重 divergence (efficiency = 3.125%)
__global__ void bad_divergence(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid % 32 == 0) {  // 每个 warp 只有 1 个 thread 执行
        data[tid] = expensive_compute();
    }
}

// 测量：
// threads_per_inst = 1
// efficiency = 1/32 = 3.125%
// 浪费 = 96.875% 计算资源

// ─────────────────────────────────────────────

// 案例 2: 中等 divergence (efficiency = 50%)
__global__ void medium_divergence(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid % 2 == 0) {  // 偶数 thread
        data[tid] = compute_even();
    } else {  // 奇数 thread
        data[tid] = compute_odd();
    }
}

// Warp 内部交替执行
// threads_per_inst = 16 (平均)
// efficiency = 16/32 = 50%

// ─────────────────────────────────────────────

// 案例 3: 无 divergence (efficiency = 100%)
__global__ void no_divergence(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int warp_id = tid / 32;
    if (warp_id % 2 == 0) {  // 整个 warp 一起判断
        data[tid] = compute_a();
    } else {
        data[tid] = compute_b();
    }
}

// 同一个 warp 的所有 thread 走相同分支
// threads_per_inst = 32
// efficiency = 100% ✓
```

**优化策略**：

```c
// 策略 1: 重组数据，避免 divergence
// ❌ Before
for (int i = 0; i < N; i++) {
    if (type[i] == 0) {
        process_type0(data[i]);
    } else {
        process_type1(data[i]);
    }
}

// ✅ After
// 1. 分离 type0 和 type1 的数据
// 2. 分别 launch kernel
kernel_type0<<<...>>>(data_type0);
kernel_type1<<<...>>>(data_type1);

// 策略 2: 使用 ballot/shuffle 避免 divergence
// 示例：在 warp 内查找满足条件的 thread
int lane = threadIdx.x % 32;
int active = (data[idx] > threshold) ? 1 : 0;
unsigned mask = __ballot_sync(0xFFFFFFFF, active);

// 只让 lane 0 做后续操作，避免 divergence
if (lane == 0) {
    int count = __popc(mask);  // 统计满足条件的数量
    // ...
}
```

##### 指标 3: Issue Efficiency (`smsp__issue_active.pct`)

```
定义：指令发射单元的活跃百分比
公式：(Issued Instructions / Possible Instructions) × 100%

解读：
- > 80%: 很好，几乎每周期都在发射指令
- 60-80%: 良好
- 40-60%: 有停顿
- < 40%: 严重停顿，需要查看 stall reasons
```

**与 ILP (Instruction-Level Parallelism) 的关系**：

```
Issue Efficiency 低的原因：

1. 指令依赖 (Data Dependency)
   a = b + c;
   d = a * 2;  // 依赖 a，无法同时发射

2. 资源冲突 (Resource Conflict)
   两条指令需要同一个功能单元

3. 内存延迟 (Long Scoreboard Stall)
   等待内存数据返回
```

**提高 Issue Efficiency**：

```c
// ❌ 低 ILP (依赖链)
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += data[i];  // 每次迭代依赖上一次的 sum
}

// 指令序列：
// load  r1, data[0]
// fadd  sum, sum, r1    ← 依赖 sum
// load  r2, data[1]
// fadd  sum, sum, r2    ← 依赖上一个 fadd
// ...

// Issue Efficiency ≈ 40-50%

// ─────────────────────────────────────────────

// ✅ 高 ILP (展开 + 独立累加器)
float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
for (int i = 0; i < N; i += 4) {
    sum0 += data[i+0];  // 4 个独立的累加器
    sum1 += data[i+1];  // 可以并行执行
    sum2 += data[i+2];
    sum3 += data[i+3];
}
float sum = sum0 + sum1 + sum2 + sum3;

// 指令可以并行发射
// Issue Efficiency ≈ 70-85%
```

##### 指标 4: Warp Stall Reasons

**最关键的指标之一！**

```bash
ncu --section WarpStateStats ./program
```

**核心指标**：

```
1. stall_long_scoreboard (长记分牌停顿)
   原因：等待内存访问 (Global Memory)
   延迟：200-400 cycles

2. stall_short_scoreboard (短记分牌停顿)
   原因：等待算术运算 (FP32, INT)
   延迟：10-20 cycles

3. stall_barrier (Barrier 停顿)
   原因：等待 __syncthreads()

4. stall_memory_dependency (内存依赖)
   原因：等待 Shared Memory/L1 访问

5. stall_not_selected (未被选中)
   原因：Occupancy 不足，没有足够的 warp 隐藏延迟
```

**专家分析法**：

```python
# NCU 输出：
stall_long_scoreboard:   65%
stall_short_scoreboard:  10%
stall_barrier:           5%
stall_not_selected:      15%
active (executing):      5%

# 推理过程：

# 1. 65% 时间在等内存 → Memory-bound ✓
# 2. 只有 5% 时间在真正计算 → 计算资源严重浪费
# 3. stall_not_selected = 15% → Occupancy 不够高

# 但要注意：
# 提高 Occupancy 能减少 stall_not_selected
# 但如果 stall_long_scoreboard 还是 65%
# → 只是更多 warp 在等内存，总时间不变

# 正确优化：
# 1. 优先解决内存问题（coalescing, cache, fusion）
# 2. 内存优化后，如果 active 还是低，再考虑 ILP
# 3. 最后才考虑 Occupancy
```

**案例分析**：

```python
# 场景 1: Shared Memory intensive kernel

ncu output:
stall_barrier:          40%
stall_memory_dependency: 30%
active:                 20%
stall_not_selected:     10%

# 分析：
# 1. 40% 时间在等 barrier → 过多的 __syncthreads()
# 2. 30% 时间等 Shared Memory → 可能有 bank conflict

# 验证：
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu

# 如果 bank_conflicts > 0:
#   → 添加 padding
# 如果 bank_conflicts = 0:
#   → 检查是否可以减少 __syncthreads() 调用
```

---

### Level 3: Detailed Metrics (1 小时+)

当 Level 1 和 Level 2 无法定位问题时，需要深入底层指标。

#### 高级内存指标

##### 1. Sector 效率

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
              l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    ./program
```

```python
# 计算：
sectors_per_request = sectors / requests

# 理论最优：
# 假设每个 warp (32 threads) 每个 thread 读 4 bytes (float)
# 总共 32 × 4 = 128 bytes
# 如果完美 coalesced，刚好 1 个 cache line (128 bytes)
# 1 个 cache line = 4 sectors (每个 sector 32 bytes)
# → sectors_per_request = 4

# 实际测量如果 > 4 → 有浪费
# 例如 = 8 → 读了 2 个 cache line，浪费 50%
```

##### 2. Transaction 分析

```bash
ncu --metrics l1tex__t_set_accesses_pipe_lsu_mem_global_op_ld.sum,\
              l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld.sum \
    ./program
```

```python
# Cache set conflict rate
conflict_rate = set_conflicts / set_accesses × 100%

# > 10% → 访问模式导致 cache thrashing
# 可能原因：
# 1. 数组大小是 cache associativity 的倍数
# 2. 多个数组地址 modulo cache_size 相同

# 解决：Padding 数组大小
```

##### 3. DRAM 效率

```bash
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
              dram__sectors_read.sum,dram__sectors_write.sum \
    ./program
```

```python
# 计算实际传输的数据量
total_dram_bytes = dram__bytes_read + dram__bytes_write

# 与理论计算的 working set 对比
# 如果 total_dram_bytes >> theoretical_working_set
# → 数据被多次读取，可能是：
#    1. 没有利用 cache
#    2. Kernel 多次读取相同数据
#    3. 需要算子融合
```

#### 计算指标深度

##### 1. FP vs INT 指令比例

```bash
ncu --metrics smsp__inst_executed_pipe_fp32.sum,\
              smsp__inst_executed_pipe_fp64.sum,\
              smsp__inst_executed_pipe_fp16.sum,\
              smsp__inst_executed_pipe_xu.sum \
    ./program
```

```python
# 分析指令分布
total = fp32 + fp64 + fp16 + int

# 如果 fp64 > 0，但硬件是 consumer GPU
# → FP64 性能是 FP32 的 1/32
# → 如果可以，换成 FP32

# 如果 fp32 很高，但是矩阵乘法
# → 考虑 FP16 Tensor Core (16x 加速)
```

##### 2. 特殊函数使用

```bash
ncu --metrics smsp__inst_executed_pipe_fma.sum,\
              smsp__inst_executed_shared_special_ops.sum \
    ./program
```

```python
# special_ops 包括：sin, cos, exp, log, sqrt, rsqrt

# 如果 special_ops 占比 > 30%
# 且使用的是精确版本 (sinf, expf)
# → 改用快速版本 (__sinf, __expf)
# → 或 nvcc -use_fast_math
# → 加速 2-3x，精度损失可接受（大多数情况）
```

---

## 指标交叉验证技巧

### 技巧 1: Memory Throughput 交叉验证

```python
# 声称：Memory-bound, Memory Throughput = 85%

# 验证 1: 计算理论带宽需求
kernel_time = 10 ms
bytes_accessed = dram__bytes_read + dram__bytes_write  # 从 NCU
bandwidth_used = bytes_accessed / kernel_time

# A100 理论峰值 = 1.5 TB/s = 1500 GB/s
# 如果 bandwidth_used = 1275 GB/s
# → 1275 / 1500 = 85% ✓ 验证通过

# 验证 2: 检查 cache hit rate
# 如果 L2 hit rate = 90%
# → 只有 10% 数据来自 DRAM
# 但 DRAM Throughput = 85% → 矛盾！
# → 可能是大量写回 DRAM

# 进一步检查：
ncu --metrics dram__bytes_write.sum
# 如果 write >> read → 确认是写回瓶颈
```

### 技巧 2: Occupancy 交叉验证

```python
# 声称：Occupancy 低导致性能差

# 验证 1: 检查 stall_not_selected
# 如果 stall_not_selected < 20%
# → Occupancy 已经足够，不是瓶颈

# 验证 2: 理论 Occupancy vs 实际
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active

# 如果理论 = 100%，实际 = 50%
# → 检查 Occupancy Limiter:
ncu --section Occupancy

# 输出可能：
# - Registers (每个 thread 用了太多寄存器)
# - Shared Memory (用了太多 Shared Memory)
# - Block Size (block 太小)
```

### 技巧 3: Compute vs Memory 交叉验证

```python
# 两个都高：SM = 85%, Memory = 85%

# 可能情况 1: 真的 Balanced（少见）
# → 接近硬件极限，优化空间小

# 可能情况 2: 测量误差或理解偏差
# → 深入看：
#    - SM Active 中有多少是真正计算？(看 Issue Efficiency)
#    - Memory Throughput 中有多少是有效数据？(看 coalescing)

# 验证：Roofline 分析
# 计算 Arithmetic Intensity = FLOPS / Bytes
# 根据 AI 判断应该是 Compute 还是 Memory bound
# 如果 AI < Ridge Point → 应该是 Memory-bound
# 但测量显示 Compute-bound → 有问题，重新分析
```

---

## 实战案例：完整分析流程

### 案例：慢速 GEMM Kernel

**Step 0: Baseline**

```bash
# 运行 kernel
./gemm

# 输出：
Kernel time: 50 ms
Matrix size: 4096 × 4096 × 4096 (FP32)
```

**Step 1: 理论分析**

```python
# 计算 FLOPS
M = N = K = 4096
FLOPS = 2 * M * N * K = 2 * 4096^3 = 137 GFLOPS

# 计算 Arithmetic Intensity
Bytes = (M*K + K*N + M*N) * 4 bytes (FP32)
      = (3 * 4096^2) * 4 = 201 MB

AI = FLOPS / Bytes = 137 / 0.2 = 685 FLOPS/Byte

# A100 Ridge Point (FP32) = 19.5 TFLOPS / 1.5 TB/s = 13 FLOPS/Byte
# 685 >> 13 → Compute-bound ✓

# 理论性能：
# A100 FP32 Peak = 19.5 TFLOPS
# 期望时间 = 137 GFLOPS / 19500 GFLOPS/s = 7 ms

# 实际：50 ms
# 效率 = 7 / 50 = 14% ← 非常差！
```

**Step 2: NCU Speed of Light**

```bash
ncu --section SpeedOfLight ./gemm
```

```
输出：
SM Throughput:          18%
Memory Throughput:      25%
```

**分析**：

```
两个都很低 → Under-utilized (资源没有充分使用)
不是 Compute-bound，也不是 Memory-bound
→ 可能是 Occupancy 或 Launch 问题
```

**Step 3: Occupancy 分析**

```bash
ncu --section Occupancy ./gemm
```

```
输出：
Theoretical Occupancy:   100%
Achieved Occupancy:      15%
Limiter:                 None (!)
```

**分析**：

```
理论 Occupancy 100%，但实际只有 15%
说明不是资源限制，而是 launch configuration 问题

检查代码：
dim3 grid(1, 1);           // ← 只有 1 个 block！
dim3 block(256, 1);

问题找到了！只启动了 1 个 block
A100 有 108 个 SM，但只用了 1 个
→ 95% 的 GPU 闲置
```

**Step 4: 修复并重新测试**

```cpp
// 修复：增加 block 数量
int blockSize = 256;
int numBlocks = (M * N + blockSize - 1) / blockSize;
dim3 grid(numBlocks / 256, 256);  // 启动更多 blocks
dim3 block(blockSize);

gemm_kernel<<<grid, block>>>(...);
```

**Step 5: 再次 Profile**

```bash
ncu --section SpeedOfLight ./gemm
```

```
新输出：
SM Throughput:          75%
Memory Throughput:      30%
Kernel time:            9 ms  (从 50ms 降到 9ms，5.5x 加速！)
```

**分析**：

```
现在是 Compute-bound ✓
75% SM 利用率，接近理论期望

但还有优化空间：
理论时间 = 7 ms
实际时间 = 9 ms
效率 = 7 / 9 = 78%

可以继续优化，但已经接近合理水平
```

**Step 6: 进一步优化 (Tensor Core)**

```python
# 如果换成 FP16 + Tensor Core:

理论峰值 (FP16 Tensor Core) = 312 TFLOPS
期望时间 = 137 / 312000 = 0.44 ms

# 使用 cuBLAS:
实际时间 ≈ 0.5 ms
效率 = 88%

# 总加速比 = 50 / 0.5 = 100x！
```

**教训**：

```
1. 理论分析很重要 (发现 14% 效率)
2. NCU Speed of Light 快速定位大方向 (Under-utilized)
3. 层层深入，找到根本原因 (Launch config)
4. 修复后验证效果 (5.5x)
5. 硬件特性利用 (Tensor Core 额外 20x)
```

---

## 常见误区与陷阱

### 误区 1: "Occupancy 越高越好"

```python
# ❌ 错误
"我的 Occupancy 只有 50%，需要提高到 100%"

# ✅ 正确
"先看 stall_not_selected:
 - 如果 < 20% → Occupancy 已经够了
 - 如果 > 40% → 才需要提高 Occupancy

 而且要权衡：
 - 提高 Occupancy 可能需要减少 registers/shared memory
 - 这可能降低单个 thread 的效率
 - 总体性能可能更差"
```

**案例**：

```c
// Kernel A: Occupancy 100%, 但用不了 Shared Memory
// Time: 10 ms

// Kernel B: Occupancy 50%, 但用 Shared Memory 缓存
// Time: 6 ms

// 结论：Kernel B 更快，虽然 Occupancy 低
```

### 误区 2: "Memory Throughput 高 = 内存优化好"

```python
# ❌ 错误
"Memory Throughput = 95%，内存访问已经优化得很好"

# ✅ 正确
"Memory Throughput 高只说明带宽饱和，可能是：
 1. 访问模式很好，高效利用带宽 ✓
 2. 访问模式很差，浪费带宽 ✗

 必须检查：
 - sectors_per_request (coalescing)
 - cache hit rate (数据复用)
 - DRAM bytes vs theoretical (是否多余访问)"
```

### 误区 3: "单看一个指标"

```python
# ❌ 错误
"L2 Hit Rate = 90%，所以不是内存问题"

# ✅ 正确
"需要综合分析：
 - L2 Hit Rate = 90%
 - 但 DRAM Throughput = 85%

 → 矛盾！如果 90% 命中 L2，为什么 DRAM 这么忙？

 查看 write 操作：
 - dram__bytes_write 很高

 → 原来是写入瓶颈，不是读取
 → 优化方向：减少写回（算子融合）"
```

### 误区 4: "Stall 一定是坏事"

```python
# ❌ 错误
"stall_long_scoreboard = 60%，需要消除"

# ✅ 正确
"对于 memory-bound kernel，stall_long_scoreboard 高是正常的
 这是在等待内存数据，无法避免

 关键是：
 - 如果有足够的 Occupancy，其他 warp 可以切换执行
 - 总体上内存延迟被隐藏

 检查：
 - stall_not_selected 是否低 (< 20%)
 - Memory Throughput 是否高 (> 80%)

 如果两个都满足 → 已经优化得很好，接受这个 stall"
```

---

## 高级技巧

### 技巧 1: Baseline Comparison

```bash
# 创建 baseline
ncu --set full -o baseline ./program

# 优化后
ncu --set full -o optimized ./program

# 对比
ncu --import baseline.ncu-rep --import optimized.ncu-rep
```

在 GUI 中可以看到所有指标的对比，快速发现优化效果。

### 技巧 2: Kernel Replay

```bash
# 捕获 kernel 参数
ncu --mode=launch-and-attach --launch-skip 10 --launch-count 1 \
    -o kernel_replay ./program

# 多次 replay 测试不同配置
# 无需重新运行整个程序
```

### 技巧 3: Python 脚本自动化分析

```python
import subprocess
import re

def parse_ncu_csv(csv_file):
    """解析 NCU CSV 输出"""
    metrics = {}
    with open(csv_file) as f:
        for line in f:
            if 'SM Throughput' in line:
                metrics['sm_throughput'] = float(re.findall(r'[\d.]+', line)[0])
            # ... 解析其他指标
    return metrics

def auto_analyze(program):
    """自动分析并给出建议"""
    # 运行 NCU
    subprocess.run(f"ncu --csv --page raw {program} > ncu.csv", shell=True)

    # 解析
    m = parse_ncu_csv("ncu.csv")

    # 决策树
    if m['sm_throughput'] > 80 and m['memory_throughput'] < 60:
        print("✓ Compute-bound")
        print("建议：检查 Tensor Core 使用、ILP、Divergence")
    elif m['sm_throughput'] < 60 and m['memory_throughput'] > 80:
        print("✓ Memory-bound")
        print("建议：检查 Coalescing、Cache Hit Rate、算子融合")
    # ...

    return m

# 使用
metrics = auto_analyze("./my_kernel")
```

### 技巧 4: 自定义 Metric Set

```bash
# 创建自定义指标集合
cat > my_metrics.txt <<EOF
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
l1tex__t_sector_hit_rate.pct
lts__t_sector_hit_rate.pct
smsp__issue_active.pct
EOF

# 使用
ncu --metrics-file my_metrics.txt ./program
```

---

## 总结：NCU 分析清单

### 第一步：Speed of Light (5 分钟)

```
□ SM Throughput: _____%
□ Memory Throughput: _____%
□ 判断：Compute / Memory / Balanced / Under-utilized
```

### 第二步：Section Analysis (20 分钟)

**如果 Memory-bound:**
```
□ L2 Hit Rate: _____%
□ L1 Hit Rate: _____%
□ Sectors per Request: _____
□ Bank Conflicts: _____
□ 结论：_____
```

**如果 Compute-bound:**
```
□ Tensor Core Usage: _____%
□ Warp Divergence Efficiency: _____%
□ Issue Efficiency: _____%
□ Stall Reasons: _____
□ 结论：_____
```

### 第三步：交叉验证

```
□ 理论计算与测量是否一致？
□ 多个指标是否相互支持？
□ 是否有反常的数据？
```

### 第四步：优化决策

```
□ 最大的瓶颈是什么？
□ 预期提升多少？
□ 优化成本是多少？
□ ROI 如何？
```

---

**记住**：NCU 是工具，不是答案。真正的专家是通过 NCU 的数据，结合硬件原理和算法理解，推导出性能的根本原因。

每个指标都在讲述性能的故事，你的任务是成为一个优秀的侦探。
