# GPU 架构适配与优化指南

## 目录
1. [NVIDIA GPU 架构演进](#nvidia-gpu-架构演进)
2. [架构特性对比](#架构特性对比)
3. [针对不同架构的优化](#针对不同架构的优化)
4. [编译选项与架构](#编译选项与架构)
5. [最佳实践](#最佳实践)

---

## NVIDIA GPU 架构演进

### 主流架构时间线

```
2012  Kepler (GTX 600/700 系列)     compute_30, compute_35
        ↓
2014  Maxwell (GTX 900 系列)        compute_50, compute_52
        ↓
2016  Pascal (GTX 1000 系列)        compute_60, compute_61
        ↓
2017  Volta (V100)                  compute_70
        ↓
2018  Turing (RTX 2000 系列)        compute_75
        ↓
2020  Ampere (A100, RTX 3000 系列)  compute_80, compute_86
        ↓
2022  Ada Lovelace (RTX 4000 系列)  compute_89
        ↓
2022  Hopper (H100)                 compute_90
        ↓
2024  Blackwell (B100)              compute_100 (预计)
```

---

## 架构特性对比

### 核心参数对比表

| 架构 | SM数量 | Tensor Core | FP32 TFLOPS | FP16 TFLOPS | 内存带宽 | L2 Cache | Shared Memory |
|------|--------|-------------|-------------|-------------|----------|----------|---------------|
| **Pascal (P100)** | 56 | ❌ | 10.6 | 21.2 | 732 GB/s | 4 MB | 64 KB/SM |
| **Volta (V100)** | 80 | ✅ Gen1 | 15.7 | 31.4 (125*) | 900 GB/s | 6 MB | 128 KB/SM |
| **Turing (RTX 2080Ti)** | 68 | ✅ Gen2 | 13.4 | 26.9 (107*) | 616 GB/s | 5.5 MB | 64 KB/SM |
| **Ampere (A100)** | 108 | ✅ Gen3 | 19.5 | 78 (312*) | 1555 GB/s | 40 MB | 164 KB/SM |
| **Hopper (H100)** | 132 | ✅ Gen4 | 67 | 134 (1979*) | 3000 GB/s | 50 MB | 228 KB/SM |

\* 使用 Tensor Cores 的峰值性能

### 关键特性

#### 1. Tensor Cores（从 Volta 开始）

**Volta (V100) - Gen1:**
- 支持 FP16 矩阵乘法
- 形状：4x4x4
- 吞吐量：~8倍于 FP32 CUDA cores

**Turing (RTX 20xx) - Gen2:**
- 新增 INT8, INT4 支持
- 改进的 FP16 性能

**Ampere (A100) - Gen3:**
- 新增 TF32 数据类型（无需代码修改）
- BF16 支持
- FP64 Tensor Cores
- 稀疏加速（2:4 结构化稀疏）

**Hopper (H100) - Gen4:**
- FP8 支持
- Transformer Engine
- DPX 指令（动态编程加速）
- Thread Block Cluster

#### 2. 内存层次变化

**Ampere 的改进：**
- L2 Cache：40 MB（V100 的 6.7 倍）
- Shared Memory：最大 164 KB/SM
- L2 Cache Residency Controls（可编程缓存）

```cpp
// Ampere: 设置 L2 cache persistence
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, cacheSize);

cudaStreamAttrValue stream_attribute;
stream_attribute.accessPolicyWindow.base_ptr = ptr;
stream_attribute.accessPolicyWindow.num_bytes = size;
stream_attribute.accessPolicyWindow.hitRatio = 1.0;
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

**Hopper 的改进：**
- 分布式 Shared Memory
- Thread Block Clusters（跨 SM 协作）
- TMA（Tensor Memory Accelerator）

#### 3. 异步操作

**Volta/Turing:**
- 基本的异步拷贝（通过 streams）

**Ampere:**
- Async Copy（`memcpy_async`）
```cpp
__global__ void async_copy_kernel(int* dst, const int* src, size_t size) {
    __shared__ int buffer[256];

    // 异步拷贝到 shared memory
    __pipeline_memcpy_async(&buffer[threadIdx.x], &src[threadIdx.x], sizeof(int));
    __pipeline_commit();
    __pipeline_wait_prior(0);

    // 使用数据
    dst[threadIdx.x] = buffer[threadIdx.x];
}
```

**Hopper:**
- TMA（硬件加速的张量拷贝）
- Asynchronous Transaction Barrier

---

## 针对不同架构的优化

### 1. Pascal 优化（GTX 1080, P100）

**特点：**
- 无 Tensor Cores
- Shared Memory 较小（64 KB）
- 良好的 FP32/FP16 性能

**优化策略：**

```cpp
// 1. 使用 FP16（2倍于 FP32）
__global__ void pascal_fp16_kernel(__half* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        __half2 val = __halves2half2(data[idx], data[idx + 1]);
        val = __hmul2(val, __float2half2_rn(2.0f));
        data[idx] = val.x;
        data[idx + 1] = val.y;
    }
}

// 2. 充分利用寄存器（255/线程）
__global__ void __launch_bounds__(256, 4)  // 提高占用率
pascal_kernel(float* data) {
    // kernel code
}

// 3. Shared Memory Tiling
#define TILE_SIZE 32  // 适配 64KB shared memory
__shared__ float tile[TILE_SIZE][TILE_SIZE];
```

### 2. Volta/Turing 优化（V100, RTX 20xx）

**特点：**
- 第一代/第二代 Tensor Cores
- 独立的 FP32 和 INT32 数据路径
- Volta: 128 KB Shared Memory

**优化策略：**

```cpp
// 1. 使用 Tensor Cores (WMMA API)
#include <mma.h>
using namespace nvcuda;

__global__ void volta_tensor_core_gemm(
    half* C, const half* A, const half* B,
    int M, int N, int K
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Load and multiply
    wmma::load_matrix_sync(a_frag, A + ..., K);
    wmma::load_matrix_sync(b_frag, B + ..., K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store
    wmma::store_matrix_sync(C + ..., c_frag, N, wmma::mem_row_major);
}

// 2. FP32/INT32 并行
__global__ void volta_mixed_compute(float* f_data, int* i_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // FP32 和 INT32 可以并行执行
        float f = f_data[idx] * 2.0f;
        int i = i_data[idx] + 1;
        f_data[idx] = f;
        i_data[idx] = i;
    }
}

// 3. 利用更大的 Shared Memory (Volta)
#define TILE_SIZE_VOLTA 48  // 可以用 128KB
__shared__ float tile[TILE_SIZE_VOLTA][TILE_SIZE_VOLTA];
```

### 3. Ampere 优化（A100, RTX 30xx）

**特点：**
- 自动 TF32 加速（FP32 代码无需修改）
- 巨大的 L2 Cache (40 MB)
- Async Copy
- 稀疏加速

**优化策略：**

```cpp
// 1. TF32 自动启用（默认）
// 对于 FP32 矩阵乘法，自动获得 ~8倍加速
// 可以通过环境变量控制：
// export NVIDIA_TF32_OVERRIDE=1  # 强制启用
// export NVIDIA_TF32_OVERRIDE=0  # 禁用

// 2. 利用异步拷贝
#include <cuda/pipeline>

__global__ void ampere_async_copy(float* dst, const float* src, int N) {
    __shared__ float buffer[256];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // 异步拷贝
    cuda::memcpy_async(buffer, src + blockIdx.x * 256, sizeof(buffer), pipe);
    pipe.producer_commit();

    // 等待完成
    pipe.consumer_wait();

    // 使用数据
    dst[blockIdx.x * 256 + threadIdx.x] = buffer[threadIdx.x] * 2.0f;
}

// 3. L2 Cache Persistence
void set_l2_persistence(void* ptr, size_t size, cudaStream_t stream) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t cacheSize = min(size, prop.l2CacheSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, cacheSize);

    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr = ptr;
    attr.accessPolicyWindow.num_bytes = size;
    attr.accessPolicyWindow.hitRatio = 1.0;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;

    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

// 4. 结构化稀疏（2:4）
// 需要使用 cuSPARSELt 库
#include <cusparseLt.h>

// 5. 更大的 Shared Memory 配置
cudaFuncSetAttribute(
    kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    164 * 1024  // 164 KB
);
```

### 4. Hopper 优化（H100）

**特点：**
- FP8 Tensor Cores
- Thread Block Clusters
- TMA（Tensor Memory Accelerator）
- Transformer Engine
- DPX 指令

**优化策略：**

```cpp
// 1. Thread Block Clusters（跨 SM 协作）
__global__ void __cluster_dims__(2, 1, 1)  // 2个 block 形成一个 cluster
hopper_cluster_kernel(float* data) {
    // 获取 cluster 信息
    dim3 cluster_dims = cluster_dims();
    dim3 cluster_id = cluster_id();

    // Cluster-wide 同步
    cluster_sync();

    // Distributed Shared Memory（跨 SM 的 shared memory）
    extern __shared__ float shared[];

    // 可以访问 cluster 内其他 block 的 shared memory
    float* remote_shared = cluster_map_shared_rank(shared, 1);  // 访问 block 1 的 shared mem
}

// 2. TMA (Tensor Memory Accelerator)
// 硬件加速的张量拷贝
__global__ void hopper_tma_kernel(float* dst, const float* src) {
    // TMA 描述符（需要在主机端创建）
    // 然后在 kernel 中使用 TMA 指令
    // 大幅降低拷贝延迟
}

// 3. FP8 Tensor Cores
// 需要使用 Transformer Engine 或 cuBLAS
#include <transformer_engine.h>

// 4. DPX 指令（动态规划加速）
// 例如：Smith-Waterman, Levenshtein Distance
// 硬件加速的动态规划操作

// 5. Warp Specialization
__global__ void hopper_warp_specialized() {
    int warp_id = threadIdx.x / 32;

    if (warp_id == 0) {
        // Producer warp: 负责加载数据
        load_data();
    } else {
        // Consumer warps: 负责计算
        compute();
    }
}
```

---

## 编译选项与架构

### 1. 基础编译

```bash
# 指定单一架构
nvcc -arch=sm_80 kernel.cu  # Ampere

# 指定多个架构（fat binary）
nvcc -arch=sm_70 -arch=sm_80 -arch=sm_90 kernel.cu

# 使用 PTX（可移植）
nvcc -arch=compute_80 -code=sm_80,compute_80 kernel.cu
```

### 2. 架构代码对照

| 架构 | -arch 选项 | GPU 型号 |
|------|-----------|---------|
| Pascal | sm_60 | P100 |
| Pascal | sm_61 | GTX 1080, GTX 1080 Ti |
| Volta | sm_70 | V100 |
| Turing | sm_75 | RTX 2080, RTX 2080 Ti, T4 |
| Ampere | sm_80 | A100 |
| Ampere | sm_86 | RTX 3090, RTX 3080 |
| Ada | sm_89 | RTX 4090, RTX 4080 |
| Hopper | sm_90 | H100 |

### 3. CMake 配置示例

```cmake
cmake_minimum_required(VERSION 3.18)
project(cuda_project CUDA CXX)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)

# 或者动态检测
include(FindCUDA/select_compute_arch)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)

# 添加可执行文件
add_executable(my_kernel kernel.cu)
target_compile_options(my_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    -Xptxas=-v
    -Xptxas=-warn-spills
>)
```

### 4. 运行时架构检测

```cpp
#include <cuda_runtime.h>
#include <iostream>

void print_device_info() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Shared Memory/Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers/Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize / (1024 * 1024) << " MB" << std::endl;
}

// 运行时分发
template<typename Kernel>
void dispatch_kernel(Kernel kernel, int major, int minor) {
    if (major == 7 && minor == 0) {
        // Volta-specific optimizations
        kernel.volta_version();
    } else if (major == 8 && minor == 0) {
        // Ampere-specific optimizations
        kernel.ampere_version();
    } else if (major == 9 && minor == 0) {
        // Hopper-specific optimizations
        kernel.hopper_version();
    } else {
        // Generic version
        kernel.generic_version();
    }
}
```

---

## 最佳实践

### 1. 多架构支持策略

```cpp
// 使用预处理器宏
#if __CUDA_ARCH__ >= 800
    // Ampere+ specific code
    #define USE_TF32
    #define USE_ASYNC_COPY
#elif __CUDA_ARCH__ >= 700
    // Volta+ specific code
    #define USE_TENSOR_CORES
#else
    // Generic code
#endif

__global__ void adaptive_kernel(float* data) {
#ifdef USE_ASYNC_COPY
    // Ampere async copy
    cuda::memcpy_async(...);
#else
    // Traditional copy
    data[threadIdx.x] = ...;
#endif
}
```

### 2. 性能可移植性

```cpp
// 使用模板和特化
template<int ARCH>
struct KernelImpl {
    __device__ static void execute(float* data) {
        // Generic implementation
    }
};

// Ampere 特化
template<>
struct KernelImpl<80> {
    __device__ static void execute(float* data) {
        // Ampere-optimized implementation
    }
};

__global__ void universal_kernel(float* data) {
#if __CUDA_ARCH__ == 800
    KernelImpl<80>::execute(data);
#else
    KernelImpl<0>::execute(data);
#endif
}
```

### 3. 资源配置自适应

```cpp
void configure_kernel_launch(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // 根据架构调整参数
    int blockSize;
    int sharedMemSize;

    if (prop.major >= 8) {  // Ampere+
        blockSize = 256;
        sharedMemSize = 128 * 1024;  // 128 KB
    } else if (prop.major >= 7) {  // Volta+
        blockSize = 256;
        sharedMemSize = 96 * 1024;   // 96 KB
    } else {  // Pascal
        blockSize = 128;
        sharedMemSize = 48 * 1024;   // 48 KB
    }

    int gridSize = (N + blockSize - 1) / blockSize;
    kernel<<<gridSize, blockSize, sharedMemSize>>>(data);
}
```

### 4. 性能测试矩阵

```python
# 测试脚本
import subprocess

architectures = [
    ('sm_70', 'V100'),
    ('sm_75', 'T4'),
    ('sm_80', 'A100'),
    ('sm_86', 'RTX3090'),
    ('sm_90', 'H100'),
]

for arch, name in architectures:
    print(f"\n=== Testing on {name} ({arch}) ===")

    # 编译
    cmd = f"nvcc -arch={arch} -O3 kernel.cu -o kernel_{arch}"
    subprocess.run(cmd, shell=True)

    # 运行测试
    subprocess.run(f"./kernel_{arch}", shell=True)
```

### 5. 特性检测工具

```cpp
#include <iostream>
#include <cuda_runtime.h>

void check_features(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "=== Device Features ===" << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Tensor Cores
    bool has_tensor_cores = (prop.major >= 7);
    std::cout << "Tensor Cores: " << (has_tensor_cores ? "YES" : "NO") << std::endl;

    // Async Copy
    bool has_async_copy = (prop.major >= 8);
    std::cout << "Async Copy: " << (has_async_copy ? "YES" : "NO") << std::endl;

    // L2 Persistence
    bool has_l2_persist = (prop.major >= 8);
    std::cout << "L2 Persistence: " << (has_l2_persist ? "YES" : "NO") << std::endl;

    // FP64 Tensor Cores
    bool has_fp64_tc = (prop.major >= 8);
    std::cout << "FP64 Tensor Cores: " << (has_fp64_tc ? "YES" : "NO") << std::endl;

    // Thread Block Clusters
    bool has_clusters = (prop.major >= 9);
    std::cout << "Thread Block Clusters: " << (has_clusters ? "YES" : "NO") << std::endl;

    // Memory info
    std::cout << "\n=== Memory Hierarchy ===" << std::endl;
    std::cout << "Shared Memory/SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Memory Bandwidth: "
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
              << " GB/s" << std::endl;
}
```

---

## 架构迁移检查清单

从老架构迁移到新架构时的检查项：

### Pascal → Volta/Turing ✓
- [ ] 添加 Tensor Core 支持（WMMA）
- [ ] 利用更大的 Shared Memory (Volta)
- [ ] 考虑 FP32/INT32 并行
- [ ] 更新编译选项 `-arch=sm_70` 或 `-arch=sm_75`

### Volta/Turing → Ampere ✓
- [ ] 启用 TF32（自动，或显式控制）
- [ ] 使用异步拷贝（`memcpy_async`）
- [ ] 配置 L2 Cache Persistence
- [ ] 考虑稀疏加速（2:4）
- [ ] 利用更大的 L2 Cache
- [ ] 更新编译选项 `-arch=sm_80` 或 `-arch=sm_86`

### Ampere → Hopper ✓
- [ ] 使用 Thread Block Clusters
- [ ] 采用 TMA 加速张量拷贝
- [ ] 考虑 FP8 Tensor Cores
- [ ] 利用 DPX 指令（如适用）
- [ ] Warp Specialization
- [ ] 更新编译选项 `-arch=sm_90`

---

## 参考资源

- [CUDA C++ Programming Guide - Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [NVIDIA GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

---

## 下一步

- 学习 [性能分析工具](../04_performance_analysis/ncu_guide.md)
- 查看 [最佳实践](best_practices.md)
- 实践不同架构的优化技巧
