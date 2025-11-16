# GPU 优化技术决策完全指南

## 目录
1. [优化决策总览](#优化决策总览)
2. [算子融合 - 何时使用](#算子融合---何时使用)
3. [ILP 优化 - 何时使用](#ilp-优化---何时使用)
4. [向量化 - 何时使用](#向量化---何时使用)
5. [Shared Memory - 何时使用](#shared-memory---何时使用)
6. [循环展开 - 何时使用](#循环展开---何时使用)
7. [Tensor Cores - 何时使用](#tensor-cores---何时使用)
8. [完整决策流程](#完整决策流程)
9. [实战案例](#实战案例)

---

## 优化决策总览

### 优化技术分类

```
GPU 优化技术
│
├─ 1. 系统级优化（影响最大，优先级最高）
│  ├─ 算子融合 (Kernel Fusion)          - 3-10x 提升
│  ├─ CUDA Graphs                       - 2-3x 提升（小kernel）
│  └─ 多流并行 (Multi-Stream)           - 1.5-2x 提升
│
├─ 2. 内存优化（Memory-bound 必做）
│  ├─ Shared Memory 缓存                - 2-5x 提升
│  ├─ 向量化访问 (float4)               - 1.5-2x 提升
│  ├─ Coalesced Access                  - 2-4x 提升
│  └─ Bank Conflict 消除                - 1.2-1.5x 提升
│
├─ 3. 计算优化（Compute-bound 必做）
│  ├─ Tensor Cores                      - 5-20x 提升
│  ├─ ILP 提升                          - 1.3-2x 提升
│  ├─ Warp Divergence 消除              - 1.5-3x 提升
│  └─ 快速数学函数                      - 1.5-3x 提升（超越函数）
│
└─ 4. 微调优化（锦上添花）
   ├─ 循环展开                          - 1.1-1.3x 提升
   ├─ Occupancy 调优                    - 1.1-1.5x 提升
   └─ 寄存器优化                        - 1.05-1.2x 提升
```

### 决策优先级

```
第一优先级：系统级优化
  → 如果有多个小 kernel → 算子融合
  → 如果 kernel 序列重复执行 → CUDA Graphs

第二优先级：根据瓶颈类型
  → Memory-bound → 内存优化
  → Compute-bound → 计算优化

第三优先级：细节优化
  → 在前两步基础上进行微调
```

---

## 算子融合 - 何时使用

### 📋 判断标准

#### ✅ 应该使用算子融合的情况

**1. 存在多个连续的 Element-wise 操作**

```python
# ❌ 融合前：3 个 kernel
x = relu(x)           # Kernel 1: 4 ms
x = x + bias         # Kernel 2: 4 ms
x = x * scale        # Kernel 3: 4 ms
# 总时间：12 ms + kernel 启动开销

# ✅ 融合后：1 个 kernel
x = fused_relu_bias_scale(x, bias, scale)
# 总时间：4 ms
# 提升：3x
```

**NCU 特征**：
```bash
ncu --section SpeedOfLight

看到：
- SM Throughput: 10-20%（计算单元空闲）
- Memory Throughput: 80-95%（内存饱和）
- Duration: 很短（< 10 ms）

同时 nsys 显示：
- 多个小 kernel 连续执行
- Kernel 之间有 gap
```

**具体判断条件**：
```
满足以下任一条件就应该融合：

1. 有 3 个以上连续的 element-wise kernel
   → 融合预期提升：2-5x

2. kernel duration < 100 μs
   → kernel 启动开销占比 > 10%
   → 融合预期提升：2-3x

3. L2 hit rate < 30% 且是连续 kernel
   → 中间结果没有复用
   → 融合预期提升：3-10x

4. 算子之间有相同的输入
   → 可以共享读取
   → 融合预期提升：1.5-3x
```

**2. 有中间结果可以消除**

```python
# ❌ 融合前
temp1 = matmul(A, B)      # 写 temp1 到 global memory
temp2 = relu(temp1)       # 读 temp1，写 temp2
output = add(temp2, bias) # 读 temp2

# ✅ 融合后
output = fused_matmul_relu_bias(A, B, bias)
# temp1, temp2 只存在于 register/shared memory
```

**NCU 特征**：
```
第一个 kernel 的 dram__bytes_write.sum
≈ 第二个 kernel 的 dram__bytes_read.sum

说明：第一个 kernel 的输出是第二个的输入
→ 可以融合，消除中间内存传输
```

**3. 小 kernel 密集执行**

```bash
# nsys 时间线显示
Kernel1 | Kernel2 | Kernel3 | Kernel4 | ...
   1ms     1ms       1ms       1ms
```

**判断条件**：
```
如果：
- 单个 kernel duration < 5 ms
- 连续执行 > 5 个 kernel
- GPU utilization < 60%（有 gap）

→ 应该融合
→ 预期提升：2-5x
```

#### ❌ 不应该使用算子融合的情况

**1. Kernel 之间有数据依赖**

```cpp
// ❌ 无法融合
x = compute1(data);
y = compute2(x);    // 依赖 x 的全部结果
z = compute3(x, y); // 依赖 x 和 y 的全部结果

// 原因：compute2 需要等待 compute1 完全结束
```

**2. Kernel 已经很大且 compute-bound**

```
如果单个 kernel：
- Duration > 50 ms
- SM Throughput > 80%
- Compute-bound

→ 不要融合（已经很高效）
→ 融合可能导致寄存器溢出，降低性能
```

**3. 融合后寄存器/Shared Memory 溢出**

```cpp
// 检查资源使用
nvcc --ptxas-options=-v kernel.cu

输出：
ptxas info    : Used 128 registers per thread
ptxas info    : Used 96 KB shared memory

如果融合后：
ptxas info    : Used 255 registers per thread  // 接近上限！
ptxas info    : Used 164 KB shared memory      // 超过限制！

→ 不要融合
→ 会导致占用率大幅下降
```

### 📊 融合效果预测

| 场景 | 融合前 | 融合后 | 提升 |
|------|--------|--------|------|
| 3个 element-wise | 12 ms | 4 ms | 3x |
| 5个 element-wise | 20 ms | 4 ms | 5x |
| MatMul + ReLU + Bias | 15 ms | 12 ms | 1.25x |
| Conv + BN + ReLU | 20 ms | 15 ms | 1.33x |
| Softmax（内部融合） | 8 ms | 2 ms | 4x |

### 💡 融合实践指南

```cpp
// 步骤 1：识别可融合的 kernel
// 使用 nsys 查看时间线
nsys profile -o timeline ./program
// 找到连续的小 kernel

// 步骤 2：编写融合 kernel
__global__ void fused_kernel(
    float* data,
    float bias,
    float scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 融合多个操作
        float val = data[idx];
        val = fmaxf(0.0f, val);  // ReLU
        val += bias;              // Add
        val *= scale;             // Scale
        data[idx] = val;
    }
}

// 步骤 3：验证性能
// Before:
ncu --section SpeedOfLight ./program_old
// After:
ncu --section SpeedOfLight ./program_fused

// 步骤 4：检查正确性
// 对比结果
```

---

## ILP 优化 - 何时使用

### 📋 判断标准

#### ✅ 应该使用 ILP 优化的情况

**1. Compute-bound 但指令发射受限**

**NCU 特征**：
```bash
ncu --section ComputeWorkloadAnalysis

看到：
SM Throughput: > 80%（计算任务重）
但是：
smsp__issue_active: < 60%（指令发射率低）
smsp__inst_executed_pipe_alu: < 70%（ALU 流水线利用率低）

原因：指令间依赖太强，无法并行发射
解决：提高 ILP
```

**判断公式**：
```python
if (SM_Throughput > 80 and issue_active < 60):
    print("需要 ILP 优化")
    print("预期提升：1.3-2x")
```

**2. 每个线程的独立操作少**

```cpp
// ❌ ILP 低
__global__ void low_ilp(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = data[idx];
        a = a * 2.0f;    // 依赖前一条
        a = a + 1.0f;    // 依赖前一条
        a = a * a;       // 依赖前一条
        data[idx] = a;   // 依赖前一条
        // 每条指令都依赖前一条，无法并行
    }
}

NCU 显示：
smsp__issue_active: 40%
```

**3. 有大量独立的数据可以处理**

```cpp
// ✅ ILP 高
__global__ void high_ilp(float* data, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        // 4 个独立的操作可以并行
        float a0 = data[idx + 0];
        float a1 = data[idx + 1];
        float a2 = data[idx + 2];
        float a3 = data[idx + 3];

        // 这些操作是独立的，可以并行执行
        a0 = a0 * 2.0f + 1.0f;
        a1 = a1 * 2.0f + 1.0f;
        a2 = a2 * 2.0f + 1.0f;
        a3 = a3 * 2.0f + 1.0f;

        data[idx + 0] = a0;
        data[idx + 1] = a1;
        data[idx + 2] = a2;
        data[idx + 3] = a3;
    }
}

NCU 显示：
smsp__issue_active: 75%（提升！）
```

#### ❌ 不应该使用 ILP 优化的情况

**1. Memory-bound kernel**

```
如果：
Memory Throughput > 80%
SM Throughput < 40%

→ 不要做 ILP 优化
→ 瓶颈在内存，不在计算
→ 应该先做内存优化
```

**2. 已经有很好的 ILP**

```
如果：
smsp__issue_active > 75%

→ 已经很好了
→ ILP 优化空间小
```

**3. 寄存器压力大**

```cpp
// 检查寄存器使用
nvcc --ptxas-options=-v kernel.cu

输出：
ptxas info : Used 200 registers per thread

如果提高 ILP（每个线程处理更多数据）：
ptxas info : Used 255 registers per thread  // 接近上限

→ 会导致 occupancy 降低
→ 得不偿失
```

### 📊 ILP 优化效果

| 场景 | ILP 前 issue_active | ILP 后 issue_active | 提升 |
|------|-------------------|-------------------|------|
| Element-wise (简单) | 45% | 72% | 1.6x |
| Element-wise (复杂) | 38% | 68% | 1.8x |
| Reduction | 52% | 75% | 1.4x |
| 已优化 MatMul | 78% | 82% | 1.05x（空间小） |

### 💡 ILP 实践指南

```cpp
// 步骤 1：确认需要 ILP 优化
ncu --section ComputeWorkloadAnalysis ./program

// 步骤 2：修改代码
// Before
__global__ void kernel_v1(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = compute(data[idx]);
        data[idx] = val;
    }
}

// After - 每个线程处理 4 个元素
__global__ void kernel_v2_ilp(float* data, int N) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (base + 3 < N) {
        // 独立的 4 个计算
        float v0 = compute(data[base + 0]);
        float v1 = compute(data[base + 1]);
        float v2 = compute(data[base + 2]);
        float v3 = compute(data[base + 3]);

        data[base + 0] = v0;
        data[base + 1] = v1;
        data[base + 2] = v2;
        data[base + 3] = v3;
    }
}

// 调整启动配置
int threads = 256;
int blocks_v1 = (N + threads - 1) / threads;
int blocks_v2 = ((N/4) + threads - 1) / threads;  // 注意除以 4

// 步骤 3：验证
ncu --metrics smsp__issue_active ./program_v2
```

---

## 向量化 - 何时使用

### 📋 判断标准

#### ✅ 应该使用向量化的情况

**1. Memory-bound 且访问已经 coalesced**

**NCU 特征**：
```bash
ncu --section MemoryWorkloadAnalysis

看到：
Memory Throughput: > 80%
l1tex__average_t_sectors_per_request: ≈ 1.0（已经 coalesced）

但是：
实际带宽 < 80% 理论带宽

→ 可以通过向量化进一步提升带宽利用率
```

**2. Element-wise 操作**

```cpp
// ✅ 适合向量化
// 每个元素的计算是独立的
C[i] = A[i] + B[i]
C[i] = relu(A[i])
C[i] = A[i] * scale + bias

// ❌ 不适合向量化
// 需要规约操作
sum += A[i]
```

**3. 数据对齐**

```cpp
// 检查数据对齐
void* ptr;
cudaMalloc(&ptr, N * sizeof(float));

// cudaMalloc 保证 256-byte 对齐
// 可以安全使用 float4

// 如果使用 cudaMallocPitch
size_t pitch;
cudaMallocPitch(&ptr, &pitch, width * sizeof(float), height);
// pitch 也是对齐的
```

**判断条件**：
```
如果满足：
1. Memory-bound
2. Element-wise 操作
3. 数据对齐
4. N % 4 == 0（或可以处理边界）

→ 应该使用 float4 向量化
→ 预期提升：1.5-2x
```

#### ❌ 不应该使用向量化的情况

**1. Compute-bound kernel**

```
如果：
SM Throughput > 80%
Memory Throughput < 60%

→ 向量化没用（瓶颈不在内存）
→ 应该做计算优化
```

**2. 数据未对齐**

```cpp
// ❌ 数据未对齐
float* data = some_pointer + 1;  // 偏移 1 个元素，未对齐

// 使用 float4 会导致 unaligned access
float4 val = reinterpret_cast<float4*>(data)[idx];  // 可能崩溃或很慢
```

**3. 复杂的数据依赖**

```cpp
// ❌ 不适合向量化
__global__ void prefix_sum(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < N) {
        data[idx] += data[idx - 1];  // 依赖前一个元素
    }
}
// 无法简单向量化
```

### 📊 向量化效果

| 场景 | 标量版本 | float4 版本 | 提升 |
|------|---------|-----------|------|
| Vector Add | 800 GB/s | 1200 GB/s | 1.5x |
| ReLU | 750 GB/s | 1150 GB/s | 1.53x |
| GELU | 650 GB/s | 1050 GB/s | 1.6x |
| Element-wise 融合 | 900 GB/s | 1400 GB/s | 1.56x |

### 💡 向量化实践指南

```cpp
// 步骤 1：检查是否适合
ncu --section MemoryWorkloadAnalysis ./program
// 确认 Memory-bound 且 coalesced

// 步骤 2：实现向量化
// Before
__global__ void kernel_scalar(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

// After
__global__ void kernel_vectorized(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx + 3 < N) {
        float4 val = reinterpret_cast<const float4*>(in)[idx];

        val.x = val.x * 2.0f + 1.0f;
        val.y = val.y * 2.0f + 1.0f;
        val.z = val.z * 2.0f + 1.0f;
        val.w = val.w * 2.0f + 1.0f;

        reinterpret_cast<float4*>(out)[idx] = val;
    }

    // 处理剩余元素
    for (int i = (N/4)*4; i < N; i++) {
        if (blockIdx.x * blockDim.x + threadIdx.x == 0 &&
            threadIdx.x < (N % 4)) {
            out[i] = in[i] * 2.0f + 1.0f;
        }
    }
}

// 调整启动配置
int vec_threads = N / 4;
int blocks = (vec_threads + 255) / 256;
kernel_vectorized<<<blocks, 256>>>(out, in, N);

// 步骤 3：验证
// 检查带宽提升
ncu --metrics dram__throughput ./program_vectorized
```

---

## Shared Memory - 何时使用

### 📋 判断标准

#### ✅ 应该使用 Shared Memory 的情况

**1. 数据有复用（同一数据被多次访问）**

**NCU 特征**：
```bash
ncu --section MemoryWorkloadAnalysis

看到：
L2 hit rate < 50%（数据没有在 L2 中复用）
但是：
算法上同一数据会被多次访问

→ 应该用 Shared Memory 缓存
```

**典型场景**：

**矩阵乘法 - Tiling**
```cpp
// C = A × B
// 每个 A 的元素被读取 N 次（B 的列数）
// 每个 B 的元素被读取 M 次（A 的行数）

// ✅ 使用 Shared Memory
__global__ void matmul_tiled(float* C, const float* A, const float* B,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 加载到 shared memory（复用 TILE_SIZE 次）
    As[ty][tx] = A[...];
    Bs[ty][tx] = B[...];
    __syncthreads();

    // 使用 shared memory 中的数据（快！）
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
}
```

**卷积**
```cpp
// 每个输入元素被多个输出元素使用（kernel size 次）
// ✅ 使用 Shared Memory 缓存输入
```

**Stencil 操作**
```cpp
// 每个元素需要访问邻居元素
// ✅ 使用 Shared Memory 缓存块
```

**2. L1 cache miss 严重**

```bash
ncu --section MemoryWorkloadAnalysis

看到：
l1tex__t_sector_hit_rate.pct < 70%

原因：L1 cache 太小，无法容纳所有数据
解决：显式使用 Shared Memory
```

**3. 需要线程间通信**

```cpp
// Block 内线程需要交换数据
// ✅ 必须使用 Shared Memory

// 例如：Transpose
__global__ void transpose(float* out, const float* in) {
    __shared__ float tile[32][33];  // +1 避免 bank conflict

    // 线程协作加载
    tile[ty][tx] = in[...];
    __syncthreads();

    // 转置后的访问模式
    out[...] = tile[tx][ty];
}
```

#### ❌ 不应该使用 Shared Memory 的情况

**1. 数据没有复用**

```cpp
// ❌ 不需要 Shared Memory
__global__ void vector_add(float* C, const float* A, const float* B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
    // 每个数据只访问一次，不需要缓存
}
```

**2. Shared Memory 会导致 Bank Conflicts**

```bash
ncu --section MemoryWorkloadAnalysis

如果使用 Shared Memory 后：
l1tex__data_bank_conflicts_pipe_lsu.sum > 1000

且：
conflict_rate > 20%

→ Shared Memory 反而降低性能
→ 需要重新设计访问模式或不用 Shared Memory
```

**3. Shared Memory 限制 Occupancy**

```bash
ncu --section Occupancy

看到：
Limiting Factor: Shared Memory
Theoretical Occupancy: 25%（因为 Shared Memory 太多）

→ 减少 Shared Memory 使用或不用
```

### 📊 Shared Memory 效果

| 场景 | 不用 Shared Memory | 用 Shared Memory | 提升 |
|------|------------------|----------------|------|
| MatMul (Naive → Tiled) | 500 GFLOPS | 2000 GFLOPS | 4x |
| Convolution | 1.2 TFLOPS | 3.5 TFLOPS | 2.9x |
| Transpose | 200 GB/s | 800 GB/s | 4x |
| Reduction | 150 GB/s | 600 GB/s | 4x |
| Vector Add | 1000 GB/s | 1000 GB/s | 1x（无用） |

### 💡 Shared Memory 实践指南

```cpp
// 步骤 1：分析数据复用
// 同一数据被访问几次？
// 如果 > 1 次，考虑 Shared Memory

// 步骤 2：计算 Shared Memory 大小
// 每个 block 需要多少？
int shared_per_block = TILE_SIZE * TILE_SIZE * sizeof(float);
// A100 每个 SM 有 164 KB

// 步骤 3：实现
__global__ void kernel_with_shared(float* data, int N) {
    __shared__ float shared[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict

    // 加载到 shared memory
    shared[ty][tx] = data[...];
    __syncthreads();  // 等待所有线程加载完成

    // 使用 shared memory
    float val = 0;
    for (int i = 0; i < TILE_SIZE; i++) {
        val += shared[ty][i] * something;
    }

    __syncthreads();  // 如果后续还要用，需要同步
}

// 步骤 4：验证
ncu --section MemoryWorkloadAnalysis ./program

// 检查：
// 1. DRAM 访问是否减少
// 2. Bank conflicts 是否为 0
// 3. Occupancy 是否受影响
```

---

## 循环展开 - 何时使用

### 📋 判断标准

#### ✅ 应该使用循环展开的情况

**1. 循环次数是编译时常量且较小**

```cpp
// ✅ 适合展开（次数小且固定）
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}

// ❌ 不适合展开（次数大）
#pragma unroll
for (int i = 0; i < 1000; i++) {  // 太多了
    sum += data[i];
}

// ❌ 不适合展开（次数动态）
for (int i = 0; i < N; i++) {  // N 是变量
    sum += data[i];
}
```

**2. 循环体简单，没有分支**

```cpp
// ✅ 适合展开
#pragma unroll
for (int i = 0; i < 4; i++) {
    result[i] = data[i] * 2.0f + 1.0f;  // 简单操作
}

// ❌ 不适合展开
#pragma unroll
for (int i = 0; i < 4; i++) {
    if (data[i] > threshold) {  // 有分支
        result[i] = expensive_compute(data[i]);
    }
}
```

**3. ILP 不足**

```bash
ncu --section ComputeWorkloadAnalysis

看到：
smsp__issue_active < 65%

且循环体有独立操作：
→ 展开可以提高 ILP
```

#### ❌ 不应该使用循环展开的情况

**1. 会导致寄存器溢出**

```bash
nvcc --ptxas-options=-v kernel.cu

Before unroll:
ptxas info : Used 80 registers

After unroll:
ptxas info : Used 220 registers  // 太多！
ptxas info : Spilled registers to local memory  // 溢出！

→ 不要展开，会降低性能
```

**2. 循环体很大**

```cpp
// ❌ 不要展开
#pragma unroll
for (int i = 0; i < 16; i++) {
    // 100 行代码
    complex_computation();
}
// 展开后代码会非常大，影响 I-cache
```

**3. 已经有很好的 ILP**

```bash
ncu --metrics smsp__issue_active

如果已经 > 75%：
→ 展开收益很小
→ 不值得
```

### 📊 循环展开效果

| 场景 | 不展开 | 展开 | 提升 |
|------|-------|-----|------|
| 小循环（4-8次） | 10 ms | 8.5 ms | 1.18x |
| 中循环（16-32次） | 20 ms | 18 ms | 1.11x |
| 大循环（>64次） | 50 ms | 52 ms | 0.96x（降低！） |
| 复杂循环体 | 30 ms | 35 ms | 0.86x（降低！） |

### 💡 循环展开实践指南

```cpp
// 步骤 1：决定是否展开
// 检查：循环次数 < 32？循环体简单？寄存器够用？

// 步骤 2：使用 #pragma unroll
// 自动展开
#pragma unroll
for (int i = 0; i < 8; i++) {
    result += data[i];
}

// 部分展开
#pragma unroll 4
for (int i = 0; i < N; i++) {
    // 每 4 次展开一次
}

// 步骤 3：或手动展开
// 自动展开
for (int i = 0; i < 4; i++) {
    sum += data[i];
}

// 手动展开
sum += data[0];
sum += data[1];
sum += data[2];
sum += data[3];

// 步骤 4：验证
nvcc --ptxas-options=-v kernel.cu
// 检查寄存器使用

ncu --metrics smsp__issue_active
// 检查是否提升
```

---

## Tensor Cores - 何时使用

### 📋 判断标准

#### ✅ 应该使用 Tensor Cores 的情况

**1. 矩阵乘法操作**

```cpp
// ✅ 适合 Tensor Cores
C = A × B  // 矩阵乘法
C = A × B + C  // 带累加的矩阵乘法

// ❌ 不适合
C = A + B  // Element-wise（不是矩阵乘法）
```

**2. 使用 FP16, BF16, TF32, INT8 数据类型**

```
Tensor Core 支持的数据类型：
- FP16 (所有 Tensor Core 代)
- BF16 (Ampere+)
- TF32 (Ampere+)
- INT8 (Turing+)
- FP64 (Ampere+)
- FP8 (Hopper+)
```

**NCU 特征**：
```bash
ncu --section ComputeWorkloadAnalysis

看到：
Compute-bound（SM > 80%）
大量 FP16/FP32 矩阵操作
但：
smsp__inst_executed_pipe_tensor = 0%（未使用 Tensor Cores）

→ 应该使用 Tensor Cores
→ 预期提升：5-20x
```

**3. 矩阵维度是 8/16/32 的倍数**

```
Tensor Core 矩阵形状：
- Volta/Turing: 16x16x16 (FP16)
- Ampere: 16x16x16 (FP16/BF16/TF32)
- Hopper: 16x16x16, 还支持其他形状

最佳：M, N, K 都是 16 的倍数
```

**4. 矩阵足够大**

```
Tensor Cores 对小矩阵效率低

推荐：
- M, N, K >= 128
- 最好 >= 512
```

#### ❌ 不应该使用 Tensor Cores 的情况

**1. 不是矩阵乘法**

```cpp
// ❌ 这些操作不能用 Tensor Cores
- Element-wise 操作
- Reduction
- Transpose
- 卷积（需要转换为 GEMM 格式）
```

**2. 矩阵太小**

```
如果 M, N, K < 64：
→ Tensor Core 启动开销 > 收益
→ 不如用普通 CUDA Cores
```

**3. 精度要求严格的 FP32**

```
TF32 虽然快，但精度低于 FP32
如果需要完整 FP32 精度：
→ 不能用 TF32 Tensor Cores
→ 用 FP32 CUDA Cores
```

### 📊 Tensor Core 效果

| 场景 | CUDA Cores | Tensor Cores | 提升 |
|------|-----------|-------------|------|
| MatMul FP16 (大) | 15 TFLOPS | 200 TFLOPS | 13x |
| MatMul TF32 (大) | 19 TFLOPS | 150 TFLOPS | 8x |
| MatMul FP32 (大) | 19 TFLOPS | 19 TFLOPS | 1x（无用） |
| MatMul FP16 (小,64x64) | 2 TFLOPS | 5 TFLOPS | 2.5x |

### 💡 Tensor Core 实践指南

```cpp
// 步骤 1：检查是否适合
// - 矩阵乘法？
// - 使用 FP16/TF32？
// - 矩阵够大？

// 步骤 2：使用 WMMA API（手动）
#include <mma.h>
using namespace nvcuda;

__global__ void matmul_wmma(
    half* C, const half* A, const half* B,
    int M, int N, int K
) {
    // 声明 fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // 初始化
    wmma::fill_fragment(c_frag, 0.0f);

    // 加载
    wmma::load_matrix_sync(a_frag, A + ..., K);
    wmma::load_matrix_sync(b_frag, B + ..., K);

    // 矩阵乘法（使用 Tensor Cores）
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 存储
    wmma::store_matrix_sync(C + ..., c_frag, N, wmma::mem_row_major);
}

// 或使用 cuBLAS（推荐）
cublasGemmEx(handle,
             CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             A, CUDA_R_16F, lda,
             B, CUDA_R_16F, ldb,
             &beta,
             C, CUDA_R_16F, ldc,
             CUDA_R_16F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);  // 使用 Tensor Cores

// 步骤 3：验证
ncu --metrics smsp__inst_executed_pipe_tensor ./program

// 应该看到：
// smsp__inst_executed_pipe_tensor > 0
// 且性能大幅提升
```

---

## 完整决策流程

### 第一步：识别瓶颈类型

```bash
ncu --section SpeedOfLight ./program
```

```
┌─────────────────────────────────────────┐
│    看 SM % 和 Memory %                   │
└─────────────────────────────────────────┘
              ↓
      ┌───────┴───────┐
      ↓               ↓
  SM > 80%        Memory > 80%
      ↓               ↓
 Compute-bound    Memory-bound
      ↓               ↓
   第二步A          第二步B
```

### 第二步A：Compute-bound 优化决策

```
Compute-bound 检测到！
│
├─ 1. 是矩阵乘法且用 FP16？
│  └─ ✅ → 使用 Tensor Cores（5-20x）
│
├─ 2. smsp__thread_inst_executed_per_inst_executed < 24？
│  └─ ✅ → 消除 Warp Divergence（1.5-3x）
│
├─ 3. smsp__issue_active < 60%？
│  └─ ✅ → 提高 ILP（1.3-2x）
│     - 每个线程处理多个数据
│     - 循环展开（如果循环小）
│
├─ 4. 有超越函数（exp, log, sin）？
│  └─ ✅ → 使用快速数学函数（1.5-3x）
│
└─ 5. 循环次数小（<16）且固定？
   └─ ✅ → 循环展开（1.1-1.3x）
```

### 第二步B：Memory-bound 优化决策

```
Memory-bound 检测到！
│
├─ 1. 有多个（>3）连续的小 kernel？
│  └─ ✅ → 算子融合（3-10x）★最优先
│
├─ 2. 同一数据被多次访问？
│  │  (L2 hit rate < 50%)
│  └─ ✅ → 使用 Shared Memory（2-5x）
│
├─ 3. l1tex__average_t_sectors_per_request > 1.5？
│  └─ ✅ → 修复 Coalesced Access（2-4x）
│
├─ 4. 已经 coalesced 但带宽 < 80%？
│  └─ ✅ → 向量化 float4（1.5-2x）
│
├─ 5. l1tex__data_bank_conflicts > 0？
│  └─ ✅ → 消除 Bank Conflicts（1.2-1.5x）
│
└─ 6. 都做完了还不够？
   └─ → 考虑算法级优化、CUDA Graphs
```

### 第三步：验证和迭代

```bash
# 优化后重新 profile
ncu --section SpeedOfLight ./program_optimized

# 对比
ncu-ui baseline.ncu-rep optimized.ncu-rep

# 如果还不够，回到第一步
```

---

## 实战案例

### 案例 1：优化 Element-wise Kernel

**初始状态**：
```bash
ncu --section SpeedOfLight ./program

SM Throughput: 8%
Memory Throughput: 92%
Duration: 15 ms
```

**分析**：Memory-bound

**优化路径**：

```
第一步：检查是否有多个 kernel
nsys profile ./program

发现：
- relu_kernel: 5 ms
- add_bias_kernel: 5 ms
- scale_kernel: 5 ms
总共 15 ms

决策：✅ 算子融合（优先级 1）

实施：
__global__ void fused_kernel(float* data, float bias, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        val = fmaxf(0.0f, val);
        val += bias;
        val *= scale;
        data[idx] = val;
    }
}

结果：
Duration: 15 ms → 5 ms
加速比：3x ✓
```

**继续优化**：

```
第二步：检查 coalesced
ncu --section MemoryWorkloadAnalysis ./program

l1tex__average_t_sectors_per_request = 1.02 ✓（已经 coalesced）

第三步：检查带宽
实际带宽：800 GB/s
理论带宽：1555 GB/s
利用率：51%

决策：✅ 向量化（优先级 2）

实施 float4：
（代码见前面向量化部分）

结果：
Duration: 5 ms → 3.2 ms
总加速比：15/3.2 = 4.7x ✓
```

### 案例 2：优化 MatMul Kernel

**初始状态**：
```bash
ncu --section SpeedOfLight ./matmul

SM Throughput: 45%
Memory Throughput: 75%
Duration: 20 ms
```

**分析**：Balanced，但都不高

**优化路径**：

```
第一步：检查 Occupancy
ncu --section Occupancy

Achieved Occupancy: 35%
Limiting Factor: Registers (200 per thread)

决策：✅ 先提高 Occupancy

实施：
__global__ void __launch_bounds__(256, 4)
matmul_kernel(...) {
    // 限制寄存器使用
}

结果：
Occupancy: 35% → 62%
Duration: 20 ms → 15 ms
```

**继续优化**：

```
第二步：检查数据复用
ncu --section MemoryWorkloadAnalysis

L2 hit rate: 25%（很低！）

决策：✅ 使用 Shared Memory Tiling

实施：
（Tiled MatMul代码见前面）

结果：
Duration: 15 ms → 6 ms
```

**最终优化**：

```
第三步：检查是否可用 Tensor Cores

条件：
- 矩阵乘法 ✓
- 可以用 FP16 ✓
- 矩阵大小：2048x2048 ✓

决策：✅ 使用 Tensor Cores

实施：
使用 cuBLAS with Tensor Cores

结果：
Duration: 6 ms → 0.8 ms
总加速比：20/0.8 = 25x ✓
```

---

## 优化决策速查表

| 场景特征 | NCU 指标 | 优化方法 | 优先级 | 预期提升 |
|---------|---------|---------|--------|---------|
| 多个小 kernel 连续 | Memory>80%, Duration<10ms | **算子融合** | ⭐⭐⭐ | 3-10x |
| 数据多次访问 | L2 hit<50% | **Shared Memory** | ⭐⭐⭐ | 2-5x |
| 矩阵乘法+FP16 | SM>80%, FP16 ops | **Tensor Cores** | ⭐⭐⭐ | 5-20x |
| 跨步访问 | sectors/request>1.5 | **修复 Coalesced** | ⭐⭐⭐ | 2-4x |
| Memory-bound+coalesced | Memory>80%, sectors=1.0 | **向量化 float4** | ⭐⭐ | 1.5-2x |
| 指令发射低 | issue_active<60% | **提高 ILP** | ⭐⭐ | 1.3-2x |
| Warp 分支多 | threads/inst<24 | **消除 Divergence** | ⭐⭐ | 1.5-3x |
| 小循环固定 | - | **循环展开** | ⭐ | 1.1-1.3x |
| Bank conflicts | conflicts>0 | **添加 padding** | ⭐⭐ | 1.2-1.5x |
| Occupancy 低 | Occupancy<40% | **调整资源** | ⭐⭐ | 1.2-2x |

**优先级说明**：
- ⭐⭐⭐：必做（影响大）
- ⭐⭐：推荐做（有明显效果）
- ⭐：可选（锦上添花）

---

## 总结

### 优化决策的黄金法则

1. **先看大的，再看小的**
   - 算子融合（3-10x）> 单个 kernel 优化（1.5-3x）

2. **先解决瓶颈，再做优化**
   - Memory-bound → 不做计算优化
   - Compute-bound → 不做向量化

3. **优先做收益大的**
   - Tensor Cores（5-20x）> ILP（1.3-2x）> 循环展开（1.1-1.3x）

4. **每次优化后都验证**
   - 防止优化适得其反

5. **不要过早优化**
   - 先 profile，确认瓶颈
   - 不要猜测

### 快速决策流程

```
1. ncu --section SpeedOfLight
   → 判断 Compute 还是 Memory bound

2. 如果 Memory-bound：
   → 先看有没有多个 kernel（融合）
   → 再看 L2 hit rate（Shared Memory）
   → 再看 coalesced（修复访问模式）
   → 最后向量化

3. 如果 Compute-bound：
   → 先看是否矩阵乘法（Tensor Cores）
   → 再看 divergence（消除分支）
   → 再看 ILP（提高并行）

4. 验证，迭代
```
