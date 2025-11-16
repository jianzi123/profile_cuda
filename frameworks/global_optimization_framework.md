# GPU 性能优化全局分析框架

> **核心思想**：系统化的方法论，从问题识别到优化验证的完整闭环，每一步都有明确的操作和判断标准。

---

## 目录

1. [优化框架总览](#优化框架总览)
2. [Phase 0: 问题定义](#phase-0-问题定义)
3. [Phase 1: 理论分析](#phase-1-理论分析)
4. [Phase 2: NCU 采样与分析](#phase-2-ncu-采样与分析)
5. [Phase 3: 优化决策](#phase-3-优化决策)
6. [Phase 4: 代码实现](#phase-4-代码实现)
7. [Phase 5: 效果验证](#phase-5-效果验证)
8. [Phase 6: 迭代优化](#phase-6-迭代优化)
9. [完整案例演示](#完整案例演示)

---

## 优化框架总览

### 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 0: 问题定义                        │
│  - 性能目标是什么？                                          │
│  - 当前性能如何？                                            │
│  - 差距在哪里？                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 1: 理论分析                          │
│  - Roofline 模型                                             │
│  - 计算理论极限                                              │
│  - 评估优化空间                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Phase 2: NCU 采样与分析                      │
│  - 运行 NCU 采样                                             │
│  - 打开 NCU-UI 分析                                          │
│  - 识别瓶颈类型                                              │
│  - 定位具体问题                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 3: 优化决策                          │
│  - 根据 NCU 数据选择优化技术                                 │
│  - 评估 ROI                                                  │
│  - 确定优先级                                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 4: 代码实现                          │
│  - 应用优化技术                                              │
│  - 保持代码正确性                                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 5: 效果验证                          │
│  - 重新 NCU 采样                                             │
│  - 对比前后数据                                              │
│  - 验证加速比                                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 6: 迭代优化                          │
│  - 达到目标？ YES → 完成                                     │
│             NO  → 回到 Phase 2，继续优化                     │
└─────────────────────────────────────────────────────────────┘
```

### 优化原则

1. **Measure, Don't Guess**（测量，不要猜）
   - 每一步都基于数据
   - 不凭感觉优化

2. **Optimize for ROI**（优化投资回报率）
   - 优先做高收益的优化
   - 避免过早优化

3. **Understand Before Optimize**（理解后再优化）
   - 先理解瓶颈原因
   - 再选择优化技术

4. **Verify Every Change**（验证每次修改）
   - 每次优化后都验证
   - 确保性能真的提升了

---

## Phase 0: 问题定义

### 目标

明确优化目标和现状，评估是否值得优化。

### 操作清单

```
□ Step 1: 定义性能目标
  问题：期望的性能是多少？
  示例：希望 kernel 时间 < 5 ms

□ Step 2: 测量当前性能
  操作：
  ```bash
  # 使用 nvprof 或简单计时
  nvprof --print-gpu-trace ./program

  # 或在代码中
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  kernel<<<grid, block>>>(...);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Kernel time: %.2f ms\n", ms);
  ```

□ Step 3: 计算性能差距
  公式：Gap = 当前时间 / 目标时间
  示例：当前 25 ms / 目标 5 ms = 5x gap

  判断：
  - Gap < 1.5x：可能不值得优化
  - Gap 1.5-3x：中等优化空间
  - Gap > 3x：大优化空间

□ Step 4: ROI 评估
  问题：优化需要多少时间？值得吗？

  简单公式：
  Value = (当前时间 - 目标时间) × 调用频率 × 重要性
  Cost = 预估开发时间
  ROI = Value / Cost

  判断：
  - ROI > 10：非常值得
  - ROI 3-10：值得
  - ROI < 3：考虑是否真的需要
```

### 输出模板

```markdown
## 性能优化问题定义

**Kernel**: vectorAdd
**当前性能**: 25 ms
**目标性能**: 5 ms
**性能差距**: 5x

**使用频率**: 训练中每 iteration 调用 100 次
**重要性**: 占总时间 40%

**ROI 评估**:
- Value: 高（占总时间 40%）
- Cost: 中（预估 2-3 天）
- ROI: 高

**结论**: ✅ 值得优化
```

---

## Phase 1: 理论分析

### 目标

在写代码/改代码之前，先通过理论分析判断优化方向和上限。

### 操作清单

```
□ Step 1: 分析 Kernel 类型
  问题：这是什么类型的 kernel？

  类型判断：
  - Element-wise (add, relu, etc.)
  - Reduction (sum, max, etc.)
  - GEMM (矩阵乘法)
  - Convolution (卷积)
  - Attention
  - 其他

  记录：__________________

□ Step 2: 计算 Arithmetic Intensity (AI)

  公式：AI = FLOPS / Bytes Accessed

  示例：Vector Add
  FLOPS = N (N 个加法)
  Bytes = 3 * N * sizeof(float) = 12N bytes (读 a, b，写 c)
  AI = N / 12N = 0.083 FLOPS/Byte

  你的 Kernel：
  FLOPS = _________________
  Bytes = _________________
  AI = _________________

□ Step 3: Roofline 分析

  硬件参数（A100 为例）：
  - FP32 Peak Compute: 19.5 TFLOPS
  - FP16 Peak Compute (Tensor Core): 312 TFLOPS
  - Peak Bandwidth: 1.5 TB/s
  - Ridge Point (FP32): 19.5 / 1.5 = 13 FLOPS/Byte
  - Ridge Point (FP16): 312 / 1.5 = 208 FLOPS/Byte

  判断：
  - AI < Ridge Point → Memory-bound
  - AI > Ridge Point → Compute-bound

  你的 Kernel：
  AI = ______
  Ridge Point = ______
  类型：Memory-bound / Compute-bound

□ Step 4: 计算理论极限

  如果 Memory-bound:
  理论时间 = Bytes / Peak Bandwidth

  如果 Compute-bound:
  理论时间 = FLOPS / Peak Compute

  你的 Kernel：
  理论时间 = _________
  当前时间 = _________
  效率 = 理论时间 / 当前时间 = ________%

  优化空间 = 当前时间 - 理论时间

□ Step 5: 设定现实目标

  理想目标：达到理论极限的 80-90%
  现实目标：达到理论极限的 60-80%

  你的目标：
  目标效率 = ______%
  目标时间 = 理论时间 / 目标效率 = ______
```

### 输出模板

```markdown
## 理论分析报告

**Kernel 类型**: Element-wise (Vector Add)

**Arithmetic Intensity**:
- FLOPS: 1,000,000 (1M additions)
- Bytes: 12,000,000 (12 MB)
- AI: 0.083 FLOPS/Byte

**Roofline 分析**:
- Ridge Point (FP32): 13 FLOPS/Byte
- 0.083 << 13 → **Memory-bound** ✓

**理论极限**:
- 理论时间: 12 MB / 1500 GB/s = 8 μs
- 当前时间: 25 ms = 25,000 μs
- 当前效率: 8 / 25,000 = 0.032% ← 非常低！

**优化空间**:
- 理论上最快: 8 μs
- 当前: 25,000 μs
- 优化空间: 3,125x ← 巨大！

**现实目标**:
- 目标效率: 60%
- 目标时间: 8 / 0.6 = 13 μs
- 从 25,000 μs 优化到 13 μs 需要 1,923x 加速

**结论**: 巨大的优化空间，主要是 Memory-bound
```

---

## Phase 2: NCU 采样与分析

### 目标

使用 NCU 采样并分析，找到具体的性能瓶颈。

### 操作清单

```
□ Step 1: 运行 NCU 采样

  命令：
  ```bash
  # 基础采样（快速）
  ncu --section SpeedOfLight -o profile ./program

  # 完整采样（详细，但慢）
  ncu --set full -o profile ./program

  # 推荐：分阶段采样
  # 第一次：Speed of Light
  ncu --section SpeedOfLight -o sol ./program

  # 第二次：根据结果，选择 Memory 或 Compute 分析
  # 如果 Memory-bound:
  ncu --section MemoryWorkloadAnalysis -o mem ./program

  # 如果 Compute-bound:
  ncu --section ComputeWorkloadAnalysis -o comp ./program
  ```

□ Step 2: 打开 NCU-UI

  命令：
  ```bash
  ncu-ui profile.ncu-rep
  ```

□ Step 3: Speed of Light 分析（5 分钟）

  位置：Details → GPU Speed Of Light

  记录：
  - SM Throughput: ______%
  - Memory Throughput: ______%
  - 瓶颈类型：________________

  判断：
  - SM > 80%, Mem < 60% → Compute-bound
  - SM < 60%, Mem > 80% → Memory-bound
  - SM > 70%, Mem > 70% → Balanced
  - SM < 50%, Mem < 50% → Under-utilized

□ Step 4: 深入分析（20 分钟）

  **如果 Memory-bound，查看**：

  位置：Details → Memory Workload Analysis

  关键指标：
  ┌────────────────────────────────────────┐
  │ 指标                  | 当前值 | 判断  │
  ├────────────────────────────────────────┤
  │ L2 Hit Rate           | ____%  | _____ │
  │ L1 Hit Rate           | ____%  | _____ │
  │ Sectors Per Request   | ___    | _____ │
  │ Bank Conflicts        | ___    | _____ │
  └────────────────────────────────────────┘

  判断标准：
  - L2 Hit Rate: >70% 好, <50% 差
  - L1 Hit Rate: >90% 好, <80% 差
  - Sectors Per Request: <1.5 好, >3 差
  - Bank Conflicts: =0 完美, >0 有问题

  **如果 Compute-bound，查看**：

  位置：Details → Compute Workload Analysis

  关键指标：
  ┌─────────────────────────────────────────┐
  │ 指标                   | 当前值 | 判断  │
  ├─────────────────────────────────────────┤
  │ Tensor Core Inst       | ___    | _____ │
  │ Threads Per Inst       | ___    | _____ │
  │ Issue Efficiency       | ____%  | _____ │
  └─────────────────────────────────────────┘

  判断标准：
  - Tensor Core Inst: =0 且是 GEMM → 问题！
  - Threads Per Inst: >29 好, <24 差 (divergence)
  - Issue Efficiency: >70% 好, <50% 差 (ILP 不足)

□ Step 5: Warp State Statistics（关键！）

  位置：Details → Warp State Statistics

  记录 Stall Reasons：
  ┌──────────────────────────────────────┐
  │ Stall Reason          | 百分比     │
  ├──────────────────────────────────────┤
  │ Long Scoreboard       | _____%     │
  │ Short Scoreboard      | _____%     │
  │ Barrier               | _____%     │
  │ Not Selected          | _____%     │
  │ Active                | _____%     │
  └──────────────────────────────────────┘

  推理：
  - Long Scoreboard 高 (>50%) → 等内存
  - Short Scoreboard 高 (>30%) → ILP 不足
  - Barrier 高 (>30%) → 过多 __syncthreads()
  - Not Selected 高 (>40%) → Occupancy 不足
  - Active 低 (<5%) → 正常，但如果 <2% → 大问题

□ Step 6: 定位具体代码（可选）

  位置：Source 页面

  操作：
  1. 点击代码行
  2. 查看右侧指标
  3. 找到红色标注的热点
  4. 记录问题代码行号：___________
```

### 输出模板

```markdown
## NCU 分析报告

**Speed of Light**:
- SM Throughput: 25%
- Memory Throughput: 88%
- **瓶颈类型: Memory-bound** ✓

**Memory Workload Analysis**:
| 指标 | 值 | 判断 | 问题 |
|-----|---|------|------|
| L2 Hit Rate | 22% | 很低 | 数据几乎没有复用 |
| L1 Hit Rate | 85% | 一般 | - |
| Sectors Per Request | 8.5 | 很差 | 带宽浪费 88% |
| Bank Conflicts | 0 | 完美 | - |

**Warp State Statistics**:
- Long Scoreboard: 85% ← 主要问题！
- Short Scoreboard: 5%
- Barrier: 2%
- Not Selected: 5%
- Active: 3%

**推理**:
1. 85% 时间在等内存 → Memory-bound 确认 ✓
2. Sectors Per Request = 8.5 → 访问模式很差，浪费 88% 带宽
3. L2 Hit Rate = 22% → 数据没有复用

**结论**:
- 主要问题：内存访问模式差（Sectors Per Request = 8.5）
- 次要问题：数据复用低（L2 Hit = 22%）
- 优化方向：1) 修复 coalescing, 2) 算子融合
```

---

## Phase 3: 优化决策

### 目标

根据 NCU 分析结果，选择合适的优化技术，评估 ROI。

### 决策树

```
                   ┌─ NCU 分析结果 ─┐
                   │                │
        ┌──────────┴────────┐       │
        │                   │       │
    Memory-bound      Compute-bound │
        │                   │       │
        ▼                   ▼       ▼

Memory-bound 决策：
├─ Q1: Sectors Per Request > 1.5?
│  └─ YES → [HIGH] Fix Memory Coalescing (预期: sectors_per_request × 加速)
│
├─ Q2: L2 Hit Rate < 50%?
│  └─ YES → [HIGH] Kernel Fusion (预期: 3-10x)
│  └─ YES → [MEDIUM] Shared Memory Tiling (预期: 2-5x)
│
├─ Q3: 已 coalesced 但带宽 < 80%?
│  └─ YES → [MEDIUM] Vectorization (float4) (预期: 1.5-2x)
│
└─ Q4: Bank Conflicts > 0?
   └─ YES → [MEDIUM] Fix Bank Conflicts (预期: 1.2-2x)

Compute-bound 决策：
├─ Q1: Tensor Core Inst = 0 且是 GEMM?
│  └─ YES → [HIGH] Use Tensor Cores (预期: 5-20x)
│
├─ Q2: Threads Per Inst < 24?
│  └─ YES → [MEDIUM] Reduce Divergence (预期: 1.5-3x)
│
├─ Q3: Issue Efficiency < 60%?
│  └─ YES → [MEDIUM] Improve ILP (预期: 1.3-2x)
│
└─ Q4: 有超越函数 (sin, exp)?
   └─ YES → [LOW] Fast Math (预期: 1.5-3x)
```

### 操作清单

```
□ Step 1: 列出所有可能的优化

  根据 NCU 分析，列出适用的优化技术：

  示例（基于上面的 NCU 报告）：
  1. Fix Memory Coalescing (Sectors Per Request = 8.5)
  2. Kernel Fusion (L2 Hit = 22%)
  3. Vectorization (如果已 coalesced)

□ Step 2: 评估每个优化的 ROI

  模板：
  ┌──────────────┬────────┬──────────┬────────┬─────┐
  │ 优化技术     │ 预期   │ 实施时间 │ 难度   │ ROI │
  ├──────────────┼────────┼──────────┼────────┼─────┤
  │ Coalescing   │ 8.5x   │ 0.5 天   │ 容易   │ ⭐⭐⭐⭐⭐ │
  │ Fusion       │ 3-5x   │ 1-2 天   │ 中等   │ ⭐⭐⭐⭐ │
  │ Vectorization│ 1.5x   │ 0.5 天   │ 容易   │ ⭐⭐⭐ │
  └──────────────┴────────┴──────────┴────────┴─────┘

  ROI = (预期加速 - 1) / 实施时间

  你的评估：
  ┌──────────────┬────────┬──────────┬────────┬─────┐
  │ 优化技术     │ 预期   │ 实施时间 │ 难度   │ ROI │
  ├──────────────┼────────┼──────────┼────────┼─────┤
  │ ____________ │ ______ │ ________ │ ______ │ ___ │
  │ ____________ │ ______ │ ________ │ ______ │ ___ │
  │ ____________ │ ______ │ ________ │ ______ │ ___ │
  └──────────────┴────────┴──────────┴────────┴─────┘

□ Step 3: 确定优先级

  规则：
  1. 先做 ROI 最高的
  2. 先做容易的（快速验证方向）
  3. 先做影响大的（Amdahl's Law）

  你的优先级：
  1. _____________________
  2. _____________________
  3. _____________________

□ Step 4: 制定实施计划

  第一次迭代：____________（预期 ____ 加速）
  第二次迭代：____________（预期 ____ 加速）
  第三次迭代：____________（预期 ____ 加速）

  总体预期：从 ____ ms → ____ ms (___x)
```

### 输出模板

```markdown
## 优化决策报告

**可行的优化技术**:

| 优化 | 预期加速 | 时间 | 难度 | ROI | 优先级 |
|-----|---------|------|------|-----|--------|
| Fix Coalescing | 8.5x | 0.5天 | 容易 | ⭐⭐⭐⭐⭐ | 1 |
| Kernel Fusion | 3-5x | 1天 | 中等 | ⭐⭐⭐⭐ | 2 |
| Vectorization | 1.5x | 0.5天 | 容易 | ⭐⭐⭐ | 3 |

**实施计划**:

**迭代 1**: Fix Memory Coalescing
- 目标：Sectors Per Request 8.5 → 1.0
- 预期：25 ms → 3 ms (8.5x)
- 时间：半天

**迭代 2**: Kernel Fusion (如果有多个 kernel)
- 目标：减少内存访问，提高 L2 Hit
- 预期：3 ms → 1 ms (3x)
- 时间：1天

**迭代 3**: Vectorization
- 目标：进一步提高带宽利用率
- 预期：1 ms → 0.7 ms (1.4x)
- 时间：半天

**总体目标**: 25 ms → 0.7 ms (35x)
```

---

## Phase 4: 代码实现

### 目标

实现优化技术，保持代码正确性。

### 操作清单

```
□ Step 1: 创建新版本

  推荐：使用版本控制
  ```bash
  # 创建新分支
  git checkout -b optimize-coalescing

  # 或者复制文件
  cp kernel.cu kernel_v2.cu
  ```

□ Step 2: 实现优化

  示例：Fix Memory Coalescing

  Before:
  ```c
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = 64;
  float val = data[tid * stride];  // 跨步访问
  ```

  After:
  ```c
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val = data[tid];  // 连续访问
  ```

  你的修改：
  ___________________________

□ Step 3: 编译测试

  ```bash
  # 编译
  nvcc -o kernel_optimized kernel.cu

  # 功能测试（确保正确性！）
  ./test_correctness
  ```

□ Step 4: 代码审查（自己检查）

  检查清单：
  - □ 逻辑正确？
  - □ 边界条件处理？
  - □ 没有引入新 bug？
  - □ 性能不会更差？

□ Step 5: 提交代码

  ```bash
  git add kernel.cu
  git commit -m "Fix memory coalescing: stride访问改为连续访问"
  ```
```

---

## Phase 5: 效果验证

### 目标

验证优化是否有效，量化性能提升。

### 操作清单

```
□ Step 1: 性能测试

  ```bash
  # 运行 benchmark
  ./benchmark

  # 或使用 nvprof
  nvprof --print-gpu-trace ./program
  ```

  记录：
  - Before: _____ ms
  - After: _____ ms
  - Speedup: _____ x

□ Step 2: NCU 验证

  ```bash
  # 优化前
  ncu -o before ./program_before

  # 优化后
  ncu -o after ./program_after

  # 对比
  ncu-ui before.ncu-rep after.ncu-rep
  ```

□ Step 3: 关键指标对比

  **优化目标指标**（以 Coalescing 为例）：

  Before:
  - Sectors Per Request: 8.5
  - DRAM Throughput: 88%
  - Duration: 25 ms

  After:
  - Sectors Per Request: 1.1 ✓
  - DRAM Throughput: 92% ✓
  - Duration: 3 ms ✓

  你的对比：
  ┌─────────────────────┬─────────┬─────────┬────────┐
  │ 指标                │ Before  │ After   │ 改善   │
  ├─────────────────────┼─────────┼─────────┼────────┤
  │ ___________________ │ _______ │ _______ │ ______ │
  │ ___________________ │ _______ │ _______ │ ______ │
  │ Duration            │ _______ │ _______ │ ______ │
  └─────────────────────┴─────────┴─────────┴────────┘

□ Step 4: 正确性验证

  重要！确保优化后结果仍然正确

  ```bash
  # 运行测试
  ./test_correctness

  # 或对比输出
  diff output_before.txt output_after.txt
  ```

  结果：□ 通过 / □ 失败

□ Step 5: 记录结果

  创建 optimization_log.md：
  ```markdown
  ## 优化记录：Fix Memory Coalescing

  **日期**: 2024-XX-XX
  **优化前**: 25 ms
  **优化后**: 3 ms
  **加速比**: 8.3x

  **修改内容**:
  - 将 stride 访问改为连续访问

  **NCU 验证**:
  - Sectors Per Request: 8.5 → 1.1 ✓

  **教训**:
  - 内存访问模式对性能影响巨大
  - 简单修改就能获得 8x 加速
  ```
```

### 输出模板

```markdown
## 优化验证报告

**优化技术**: Fix Memory Coalescing

**性能对比**:
- Before: 25 ms
- After: 3 ms
- **Speedup: 8.3x** ✓

**NCU 指标对比**:
| 指标 | Before | After | 改善 |
|-----|--------|-------|------|
| Sectors Per Request | 8.5 | 1.1 | 87% ↓ |
| DRAM Throughput | 88% | 92% | 4% ↑ |
| L2 Hit Rate | 22% | 25% | 3% ↑ |

**正确性验证**: ✅ 通过

**与预期对比**:
- 预期加速: 8.5x
- 实际加速: 8.3x
- 达成率: 98% ✓

**结论**: 优化成功，符合预期
```

---

## Phase 6: 迭代优化

### 目标

评估是否达到目标，决定是否继续优化。

### 决策流程

```
┌─────────────────────────────────────┐
│ 当前性能: 3 ms                      │
│ 目标性能: 5 ms                      │
│ 理论极限: 8 μs                      │
└──────────────┬──────────────────────┘
               │
               ▼
        是否达到目标？
               │
       ┌───────┴────────┐
       │                │
      YES              NO
       │                │
       ▼                ▼
   ┌────────┐      是否接近极限？
   │ 完成！ │           │
   └────────┘   ┌───────┴────────┐
               YES              NO
                │                │
                ▼                ▼
          可以接受，       继续优化
          考虑算法级        │
          优化            ▼
                     回到 Phase 2
                     (新一轮 NCU 分析)
```

### 操作清单

```
□ Step 1: 评估当前状态

  当前性能: _____ ms
  目标性能: _____ ms
  理论极限: _____ ms

  当前效率 = 理论极限 / 当前性能 = _____%
  目标效率 = 理论极限 / 目标性能 = _____%

□ Step 2: 决策

  判断：
  - □ 已达到目标 → 完成！
  - □ 接近理论极限 (>60%) → 考虑算法级优化
  - □ 还有优化空间 (<60%) → 继续迭代

□ Step 3: 如果继续优化

  下一步优化：_______________
  预期加速：_______________
  实施时间：_______________

  → 回到 Phase 2，重新 NCU 分析

□ Step 4: 如果完成

  总结：
  - 初始性能：_____ ms
  - 最终性能：_____ ms
  - 总加速比：_____ x
  - 优化轮次：_____ 次
  - 总耗时：_____ 天
```

---

## 完整案例演示

### 案例：优化 Vector Add Kernel

#### Phase 0: 问题定义

```
Kernel: vectorAdd
当前性能: 25 ms (N = 100M elements)
目标性能: < 5 ms
性能差距: 5x
ROI: 高（占训练时间 40%）
结论: 值得优化 ✓
```

#### Phase 1: 理论分析

```
类型: Element-wise
FLOPS: 100M
Bytes: 100M × 3 × 4 = 1.2 GB
AI: 100M / 1.2GB = 0.083 FLOPS/Byte

Ridge Point (A100): 13 FLOPS/Byte
0.083 << 13 → Memory-bound ✓

理论极限:
时间 = 1.2 GB / 1.5 TB/s = 0.8 ms
当前: 25 ms
效率: 0.8 / 25 = 3.2%
优化空间: 31x
```

#### Phase 2: NCU 分析

```
Speed of Light:
- SM: 25%
- Memory: 88%
→ Memory-bound ✓

Memory Analysis:
- L2 Hit: 22% (差)
- Sectors Per Request: 8.5 (很差！)
- Bank Conflicts: 0

Warp State:
- Long Scoreboard: 85% (等内存)

结论:
主要问题: Sectors Per Request = 8.5
次要问题: L2 Hit = 22%
```

#### Phase 3: 优化决策

```
优化计划:
1. Fix Coalescing (预期 8.5x, 0.5天, ⭐⭐⭐⭐⭐)
2. Kernel Fusion (预期 3x, 1天, ⭐⭐⭐⭐) - 如果有多个 kernel
3. Vectorization (预期 1.5x, 0.5天, ⭐⭐⭐)

优先级: 1 → 2 → 3
```

#### Phase 4: 代码实现

```c
// 迭代 1: Fix Coalescing

// Before
__global__ void vectorAdd_v0(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = 64;
    if (tid < N / stride) {
        c[tid * stride] = a[tid * stride] + b[tid * stride];
    }
}

// After
__global__ void vectorAdd_v1(float* a, float* b, float* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];  // 连续访问
    }
}
```

#### Phase 5: 效果验证

```
性能对比:
- Before: 25 ms
- After: 3 ms
- Speedup: 8.3x ✓

NCU 验证:
- Sectors Per Request: 8.5 → 1.1 ✓
- Duration: 25 ms → 3 ms ✓

正确性: ✅ 通过
```

#### Phase 6: 迭代优化

```
当前: 3 ms
目标: 5 ms
已达到目标! ✓

但理论极限是 0.8 ms，还有优化空间

下一步: Vectorization
预期: 3 ms → 2 ms (1.5x)
```

**继续优化...**

```c
// 迭代 2: Vectorization

__global__ void vectorAdd_v2(float4* a, float4* b, float4* c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N / 4) {
        c[tid] = make_float4(
            a[tid].x + b[tid].x,
            a[tid].y + b[tid].y,
            a[tid].z + b[tid].z,
            a[tid].w + b[tid].w
        );
    }
}

// 结果: 3 ms → 2 ms (1.5x) ✓
```

**最终结果**:

```
初始: 25 ms
迭代1 (Coalescing): 3 ms (8.3x)
迭代2 (Vectorization): 2 ms (1.5x)

总加速比: 12.5x
当前效率: 0.8 / 2 = 40%
结论: 接近合理水平，可以停止
```

---

## 总结：优化框架检查清单

打印此清单，每次优化时使用：

```
□ Phase 0: 问题定义
  □ 定义性能目标
  □ 测量当前性能
  □ 计算差距
  □ 评估 ROI

□ Phase 1: 理论分析
  □ 识别 Kernel 类型
  □ 计算 AI
  □ Roofline 分析
  □ 计算理论极限
  □ 设定现实目标

□ Phase 2: NCU 分析
  □ 运行 NCU 采样
  □ Speed of Light
  □ 深入分析 (Memory/Compute)
  □ Warp State Statistics
  □ 定位问题代码

□ Phase 3: 优化决策
  □ 列出可行优化
  □ 评估 ROI
  □ 确定优先级
  □ 制定实施计划

□ Phase 4: 代码实现
  □ 创建新版本
  □ 实现优化
  □ 编译测试
  □ 代码审查

□ Phase 5: 效果验证
  □ 性能测试
  □ NCU 验证
  □ 指标对比
  □ 正确性验证
  □ 记录结果

□ Phase 6: 迭代优化
  □ 评估当前状态
  □ 决策下一步
  □ 继续或完成
```

---

**记住**：优化是一个迭代过程，每一步都要：
1. 基于数据决策
2. 量化预期效果
3. 验证实际结果
4. 记录经验教训

这个框架确保你的优化是系统化、可复现、可验证的！
