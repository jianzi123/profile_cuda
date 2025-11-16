# 全局视角的算子性能分析与优化

## 目录
1. [全局优化概述](#全局优化概述)
2. [端到端性能分析](#端到端性能分析)
3. [算子融合策略](#算子融合策略)
4. [内存复用与优化](#内存复用与优化)
5. [系统级性能分析](#系统级性能分析)
6. [完整优化工作流](#完整优化工作流)

---

## 全局优化概述

### 为什么需要全局视角？

单个 kernel 的优化是微观层面，但在实际应用中：

```
End-to-End Performance ≠ Sum(Individual Kernel Performance)
```

**系统级开销：**
- Kernel 启动延迟（~5-10 μs）
- 内存传输时间
- CPU-GPU 同步
- 算子间数据依赖
- 内存分配/释放

**全局优化关注：**
1. 算子间的数据流
2. 内存使用模式
3. 并行执行机会
4. 算子融合可能性
5. 端到端吞吐量

### 优化层次

```
┌─────────────────────────────────────┐
│ 1. Application Level (应用层)       │  ← 算法选择、数据结构
├─────────────────────────────────────┤
│ 2. Framework Level (框架层)         │  ← 图优化、算子融合
├─────────────────────────────────────┤
│ 3. Kernel Level (算子层)            │  ← 单个 kernel 优化
├─────────────────────────────────────┤
│ 4. Instruction Level (指令层)       │  ← SASS 优化、寄存器分配
└─────────────────────────────────────┘
```

我们需要在**所有层次**进行优化！

---

## 端到端性能分析

### 1. 使用 Nsight Systems 全局分析

**与 NCU 的区别：**
- **Nsight Systems**：宏观，看时间线，找全局瓶颈
- **Nsight Compute**：微观，看单个 kernel，找细节问题

```bash
# 使用 Nsight Systems 收集整体 trace
nsys profile -o timeline \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    python train.py

# 查看结果
nsys-ui timeline.qdrep
```

**在时间线中查看：**

```
CPU Timeline:
  ████ Python ████ CUDA API ████ Python ████

GPU Timeline:
  Kernel1 ─┐
           │ Gap! (浪费)
           └─ Kernel2 ──┐
                        │ Memory Copy (可以重叠？)
                        └─ Kernel3

Memory Timeline:
  H→D Copy ──── D→H Copy ────
```

**寻找的问题：**
1. **GPU Idle Time**：GPU 空闲，CPU 在干什么？
2. **Kernel Launch Overhead**：太多小 kernel
3. **Memory Transfer Bottleneck**：CPU-GPU 传输过多
4. **Synchronization Points**：不必要的同步
5. **CPU Bottleneck**：CPU 跟不上 GPU

### 2. NVTX 标记关键区域

```cpp
#include <nvtx3/nvToolsExt.h>

void forward_pass() {
    nvtxRangePushA("Forward Pass");

    nvtxRangePushA("Conv1");
    conv1_kernel<<<grid, block>>>(data);
    nvtxRangePop();

    nvtxRangePushA("ReLU");
    relu_kernel<<<grid, block>>>(data);
    nvtxRangePop();

    nvtxRangePushA("Pool");
    pool_kernel<<<grid, block>>>(data);
    nvtxRangePop();

    nvtxRangePop();
}
```

**Python 版本：**
```python
import torch
from torch.cuda import nvtx

def forward(x):
    with nvtx.range("Forward Pass"):
        with nvtx.range("Conv1"):
            x = conv1(x)
        with nvtx.range("ReLU"):
            x = relu(x)
        with nvtx.range("Pool"):
            x = pool(x)
    return x
```

在 Nsight Systems 中可以看到这些标记，快速定位问题区域。

### 3. 端到端性能指标

```python
import torch
import time

def benchmark_e2e(model, input_data, num_iterations=100):
    """端到端性能测试"""

    # Warmup
    for _ in range(10):
        _ = model(input_data)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        output = model(input_data)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / num_iterations * 1000  # ms
    throughput = num_iterations / (end - start)  # samples/s

    print(f"Avg Latency: {avg_time:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/s")

    return avg_time, throughput
```

### 4. 分析 CPU-GPU 交互

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
    with_flops=True,
) as prof:
    model(input_data)

# 打印报告
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=10
))

# 导出 Chrome trace
prof.export_chrome_trace("trace.json")
# 在 chrome://tracing 中查看
```

---

## 算子融合策略

### 1. 为什么需要算子融合？

**问题：多个小 kernel**
```python
# 三个独立的 kernel
x = relu(x)            # Kernel 1: 读 x, 写 x
x = x + bias          # Kernel 2: 读 x, 写 x
x = dropout(x)        # Kernel 3: 读 x, 写 x
```

**开销：**
- 3 次 kernel 启动（~15-30 μs）
- 3 次全局内存读写
- 中间结果存储

**融合后：**
```python
# 单个融合 kernel
x = fused_relu_bias_dropout(x, bias)  # 只读写一次！
```

**收益：**
- 1 次 kernel 启动
- 1 次全局内存读写
- 减少中间存储

### 2. Element-wise 算子融合

```cpp
// 融合前：3 个 kernel
__global__ void relu_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void add_bias_kernel(float* data, float bias, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += bias;
    }
}

__global__ void scale_kernel(float* data, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= scale;
    }
}

// 融合后：1 个 kernel
__global__ void fused_relu_bias_scale(float* data, float bias, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        val = fmaxf(0.0f, val);  // ReLU
        val += bias;              // Add bias
        val *= scale;             // Scale
        data[idx] = val;
    }
}
```

**性能提升：** 通常 2-5 倍

### 3. Triton 自动融合

Triton 的 JIT 编译器可以自动融合某些操作：

```python
import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    x_ptr, bias_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 一次性完成多个操作（Triton 自动优化）
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # 融合操作
    out = tl.where(x > 0, x, 0.0)  # ReLU
    out = out + bias                # Add
    out = out * 2.0                 # Scale

    tl.store(out_ptr + offsets, out, mask=mask)
```

### 4. 更复杂的融合：LayerNorm + Linear

```cpp
// 融合 LayerNorm 和 Linear
__global__ void fused_layernorm_linear(
    const float* input,      // [B, N]
    const float* gamma,      // [N]
    const float* beta,       // [N]
    const float* weight,     // [N, M]
    const float* bias,       // [M]
    float* output,           // [B, M]
    int B, int N, int M
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    // 第一步：LayerNorm
    __shared__ float s_mean, s_var;

    // 计算均值
    float sum = 0.0f;
    for (int i = col; i < N; i += blockDim.x) {
        sum += input[row * N + i];
    }
    sum = warp_reduce_sum(sum);
    if (col == 0) s_mean = sum / N;
    __syncthreads();

    // 计算方差
    float var = 0.0f;
    for (int i = col; i < N; i += blockDim.x) {
        float diff = input[row * N + i] - s_mean;
        var += diff * diff;
    }
    var = warp_reduce_sum(var);
    if (col == 0) s_var = var / N;
    __syncthreads();

    // 归一化并存储到 shared memory
    __shared__ float normalized[1024];
    for (int i = col; i < N; i += blockDim.x) {
        float val = input[row * N + i];
        normalized[i] = (val - s_mean) * rsqrtf(s_var + 1e-5) * gamma[i] + beta[i];
    }
    __syncthreads();

    // 第二步：Linear（直接使用 shared memory 中的 normalized 数据）
    for (int j = col; j < M; j += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += normalized[k] * weight[k * M + j];
        }
        output[row * M + j] = sum + bias[j];
    }
}
```

**融合收益：**
- 避免 LayerNorm 的中间结果写回全局内存
- 利用 shared memory 缓存归一化后的数据
- 减少一次 kernel 启动

### 5. PyTorch JIT Fusion

```python
import torch

# 方法 1: TorchScript
@torch.jit.script
def fused_ops(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = torch.relu(x)
    x = x + bias
    x = x * 2.0
    return x

# PyTorch 会自动融合这些操作

# 方法 2: torch.compile (PyTorch 2.0+)
@torch.compile
def fused_ops_v2(x, bias):
    x = torch.relu(x)
    x = x + bias
    x = x * 2.0
    return x
```

---

## 内存复用与优化

### 1. 内存池（Memory Pool）

```cpp
#include <cuda_runtime.h>

class MemoryPool {
private:
    std::vector<void*> pool;
    std::vector<size_t> sizes;

public:
    void* allocate(size_t bytes) {
        // 查找可复用的内存块
        for (size_t i = 0; i < pool.size(); i++) {
            if (sizes[i] >= bytes && !in_use[i]) {
                in_use[i] = true;
                return pool[i];
            }
        }

        // 如果没有，分配新的
        void* ptr;
        cudaMalloc(&ptr, bytes);
        pool.push_back(ptr);
        sizes.push_back(bytes);
        in_use.push_back(true);
        return ptr;
    }

    void release(void* ptr) {
        for (size_t i = 0; i < pool.size(); i++) {
            if (pool[i] == ptr) {
                in_use[i] = false;
                return;
            }
        }
    }

    ~MemoryPool() {
        for (auto ptr : pool) {
            cudaFree(ptr);
        }
    }
};
```

**PyTorch 内置：**
```python
import torch

# PyTorch 自动使用内存池
# 查看内存使用
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())

# 清空缓存（如果需要）
torch.cuda.empty_cache()
```

### 2. In-place 操作

```python
# ❌ 非 in-place：创建新的 tensor
x = x + 1

# ✅ In-place：复用内存
x.add_(1)

# 常见 in-place 操作
x.relu_()
x.mul_(2.0)
x.add_(bias)
```

**CUDA 版本：**
```cpp
// In-place ReLU
__global__ void relu_inplace(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = fmaxf(0.0f, data[idx]);  // 直接修改
    }
}
```

### 3. 内存重用分析

```python
# 使用 Nsight Systems 查看内存使用模式
# nsys profile --trace=cuda --cuda-memory-usage=true python script.py

# 或使用 PyTorch Profiler
with torch.profiler.profile(
    profile_memory=True,
    record_shapes=True,
) as prof:
    model(input_data)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
```

### 4. Workspace 复用

```cpp
// 多个算子共享临时工作空间
void* workspace = nullptr;
size_t workspace_size = 0;

// 计算所需的最大 workspace
size_t conv1_ws = get_conv_workspace_size(conv1);
size_t conv2_ws = get_conv_workspace_size(conv2);
workspace_size = max(conv1_ws, conv2_ws);

cudaMalloc(&workspace, workspace_size);

// 复用 workspace
conv1_kernel<<<...>>>(data1, workspace);
conv2_kernel<<<...>>>(data2, workspace);  // 复用同一块内存

cudaFree(workspace);
```

---

## 系统级性能分析

### 1. 分析整个训练/推理流程

```python
import torch
from torch.profiler import profile, ProfilerActivity

class Model:
    def forward(self, x):
        # Model implementation
        pass

def analyze_pipeline():
    model = Model()
    input_data = torch.randn(32, 3, 224, 224, device='cuda')

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            output = model.forward(input_data)
            loss = output.sum()
            loss.backward()

    # 分析结果
    print("=== Top CUDA Kernels ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    print("\n=== Memory Usage ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))

    # 查看 CPU 时间
    print("\n=== CPU Time ===")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    ))
```

### 2. 多流并行

```python
import torch

# 创建多个 CUDA streams
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# 并行执行
with torch.cuda.stream(stream1):
    output1 = model1(input1)

with torch.cuda.stream(stream2):
    output2 = model2(input2)

# 等待所有 stream 完成
torch.cuda.synchronize()
```

**CUDA 版本：**
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 在不同 stream 中并行执行
kernel1<<<grid, block, 0, stream1>>>(data1);
kernel2<<<grid, block, 0, stream2>>>(data2);

// 异步拷贝也可以并行
cudaMemcpyAsync(d_data1, h_data1, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_data1);
cudaMemcpyAsync(h_result1, d_result1, size, cudaMemcpyDeviceToHost, stream1);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

### 3. 重叠计算与通信

```cpp
// 将数据分块，重叠拷贝和计算
const int num_chunks = 4;
const int chunk_size = N / num_chunks;

cudaStream_t streams[num_chunks];
for (int i = 0; i < num_chunks; i++) {
    cudaStreamCreate(&streams[i]);
}

for (int i = 0; i < num_chunks; i++) {
    int offset = i * chunk_size;

    // 异步拷贝
    cudaMemcpyAsync(
        d_data + offset,
        h_data + offset,
        chunk_size * sizeof(float),
        cudaMemcpyHostToDevice,
        streams[i]
    );

    // 计算（与下一个 chunk 的拷贝重叠）
    kernel<<<grid, block, 0, streams[i]>>>(d_data + offset, chunk_size);

    // 拷贝回主机
    cudaMemcpyAsync(
        h_result + offset,
        d_result + offset,
        chunk_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        streams[i]
    );
}

// 等待所有完成
for (int i = 0; i < num_chunks; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
}
```

### 4. 图优化（CUDA Graphs）

```cpp
#include <cuda_runtime.h>

cudaGraph_t graph;
cudaGraphExec_t instance;

// 开始捕获
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 执行一系列操作（这些会被记录到 graph 中）
kernel1<<<grid, block, 0, stream>>>(data);
kernel2<<<grid, block, 0, stream>>>(data);
kernel3<<<grid, block, 0, stream>>>(data);

// 结束捕获
cudaStreamEndCapture(stream, &graph);

// 实例化 graph
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

// 重复执行（开销极低！）
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);
}

cudaStreamSynchronize(stream);
```

**PyTorch 版本：**
```python
import torch

# 创建 CUDA graph
g = torch.cuda.CUDAGraph()

# Static input
static_input = torch.randn(32, 3, 224, 224, device='cuda')
static_output = torch.empty(32, 1000, device='cuda')

# Warmup
for _ in range(3):
    static_output = model(static_input)

# 捕获 graph
with torch.cuda.graph(g):
    static_output = model(static_input)

# 重复执行（非常快！）
for _ in range(1000):
    g.replay()
```

**性能提升：** 对于小 kernel 密集的工作负载，可以提升 2-3 倍

---

## 完整优化工作流

### 第一阶段：建立基线

```bash
# 1. 端到端性能测试
python benchmark.py --model resnet50 --batch-size 32

# 2. Nsight Systems 全局 profile
nsys profile -o baseline \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    python benchmark.py

# 3. 查看时间线，找到热点区域
nsys-ui baseline.qdrep
```

### 第二阶段：定位瓶颈

```bash
# 在 Nsight Systems 中识别：
# - 最耗时的 kernel
# - GPU idle 时间
# - 内存拷贝瓶颈
# - CPU 瓶颈

# 对热点 kernel 进行详细分析
ncu --set full -o kernel_profile \
    --kernel-name <bottleneck_kernel> \
    python benchmark.py

ncu-ui kernel_profile.ncu-rep
```

### 第三阶段：优化策略

根据瓶颈类型选择策略：

```python
def optimize_pipeline(bottleneck_type):
    if bottleneck_type == "many_small_kernels":
        # → 算子融合
        return fuse_kernels()

    elif bottleneck_type == "memory_bound":
        # → 内存优化
        return optimize_memory_access()

    elif bottleneck_type == "cpu_bottleneck":
        # → 减少 CPU-GPU 同步
        return use_cuda_graphs()

    elif bottleneck_type == "gpu_idle":
        # → 增加并行度
        return use_multiple_streams()

    elif bottleneck_type == "memory_copy":
        # → 重叠计算与通信
        return overlap_compute_copy()
```

### 第四阶段：验证改进

```python
import torch
import time

def compare_performance(baseline_fn, optimized_fn, input_data):
    """对比优化前后的性能"""

    # Baseline
    start = time.time()
    for _ in range(100):
        baseline_fn(input_data)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100 * 1000

    # Optimized
    start = time.time()
    for _ in range(100):
        optimized_fn(input_data)
    torch.cuda.synchronize()
    optimized_time = (time.time() - start) / 100 * 1000

    speedup = baseline_time / optimized_time

    print(f"Baseline:   {baseline_time:.2f} ms")
    print(f"Optimized:  {optimized_time:.2f} ms")
    print(f"Speedup:    {speedup:.2f}x")

    return speedup
```

### 第五阶段：持续监控

```python
#!/usr/bin/env python3
"""性能回归测试"""

import torch
import json
import subprocess
from datetime import datetime

def run_benchmark():
    """运行性能测试并记录结果"""
    # ... benchmark code

    result = {
        "timestamp": datetime.now().isoformat(),
        "latency_ms": latency,
        "throughput": throughput,
        "gpu_util": gpu_util,
        "memory_usage": memory_usage,
    }

    return result

def check_regression(current, baseline, threshold=0.1):
    """检查性能回归"""
    if current["latency_ms"] > baseline["latency_ms"] * (1 + threshold):
        print(f"⚠️  Performance regression detected!")
        print(f"   Baseline: {baseline['latency_ms']:.2f} ms")
        print(f"   Current:  {current['latency_ms']:.2f} ms")
        return True
    return False

# 主循环
baseline = load_baseline()
while True:
    current = run_benchmark()

    if check_regression(current, baseline):
        # 触发告警
        send_alert()

    save_result(current)
    time.sleep(3600)  # 每小时检查一次
```

---

## 最佳实践总结

### 优化检查清单

#### 算子级别 ✓
- [ ] 使用 Nsight Compute 分析单个 kernel
- [ ] 优化内存访问模式（coalesced access）
- [ ] 提高占用率
- [ ] 减少 warp divergence
- [ ] 使用 Tensor Cores（如果适用）

#### 系统级别 ✓
- [ ] 使用 Nsight Systems 分析全局性能
- [ ] 识别并优化 GPU idle 时间
- [ ] 算子融合减少 kernel 启动开销
- [ ] 使用 CUDA Graphs 降低启动延迟
- [ ] 多流并行提高吞吐量

#### 内存级别 ✓
- [ ] 使用内存池避免频繁分配
- [ ] In-place 操作减少内存使用
- [ ] 重叠计算与通信
- [ ] Workspace 复用

#### 应用级别 ✓
- [ ] 选择合适的算法
- [ ] 批处理提高吞吐量
- [ ] 数据预处理放在 CPU
- [ ] 异步数据加载

---

## 工具链总结

```
性能分析工具链：

1. 宏观分析（全局视角）
   ├─ Nsight Systems：时间线、CPU-GPU 交互
   ├─ PyTorch Profiler：Python 层性能
   └─ NVTX：自定义标记

2. 微观分析（Kernel 级别）
   ├─ Nsight Compute：详细指标、Roofline
   ├─ nvprof（已废弃）
   └─ cuda-memcheck：内存错误检查

3. 代码分析
   ├─ Nsight Compute：源代码级分析
   ├─ cuobjdump：查看 PTX/SASS
   └─ nvdisasm：反汇编

4. 实时监控
   ├─ nvidia-smi：GPU 利用率
   ├─ nvtop：交互式监控
   └─ dcgm：数据中心 GPU 管理
```

---

## 案例研究：优化一个完整的训练循环

```python
# === 优化前 ===
def train_loop_baseline(model, dataloader, optimizer):
    for batch in dataloader:
        # CPU-GPU 拷贝（同步点）
        inputs = batch['image'].cuda()
        labels = batch['label'].cuda()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Nsight Systems 显示：
# - 大量 H2D 拷贝时间
# - GPU 在等待数据时 idle
# - 小 kernel 启动开销大

# === 优化后 ===
def train_loop_optimized(model, dataloader, optimizer):
    # 1. 异步数据加载
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,  # ← 启用 pinned memory
        prefetch_factor=2,
    )

    # 2. 使用 AMP 混合精度
    scaler = torch.cuda.amp.GradScaler()

    # 3. 使用 CUDA Graph
    static_input = torch.randn(32, 3, 224, 224, device='cuda')
    static_label = torch.randint(0, 1000, (32,), device='cuda')

    # Warmup
    for _ in range(3):
        with torch.cuda.amp.autocast():
            output = model(static_input)
            loss = criterion(output, static_label)

    # Capture
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with torch.cuda.amp.autocast():
            static_output = model(static_input)
            static_loss = criterion(static_output, static_label)
        scaler.scale(static_loss).backward()

    # Training loop
    for batch in dataloader:
        # 拷贝数据到静态buffer
        static_input.copy_(batch['image'])
        static_label.copy_(batch['label'])

        # Replay graph（非常快！）
        g.replay()

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

# 性能提升：2-4倍！
```

---

## 总结

全局优化的核心思想：

1. **从宏观到微观**：先用 Nsight Systems 找大问题，再用 NCU 优化细节
2. **系统思维**：不只看 kernel，要看整个数据流
3. **权衡取舍**：有时候"更慢"的算法配合更好的系统优化反而更快
4. **持续监控**：性能优化不是一次性的，需要持续关注

**记住：**
```
最快的代码是不运行的代码
次快的代码是只运行一次的代码
第三快的代码是并行运行的代码
```

---

## 参考资源

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
