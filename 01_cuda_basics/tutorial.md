# CUDA 算子编写完整教程

## 目录
1. [CUDA 编程模型](#cuda-编程模型)
2. [第一个 CUDA Kernel](#第一个-cuda-kernel)
3. [内存层次结构](#内存层次结构)
4. [线程组织与索引](#线程组织与索引)
5. [完整示例](#完整示例)

---

## CUDA 编程模型

### 核心概念

CUDA 是 NVIDIA 提供的并行计算平台和编程模型，允许开发者利用 GPU 进行通用计算。

#### 1. 线程层次结构

```
Grid (网格)
├── Block 0 (线程块)
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
├── Block 1
│   ├── Thread 0
│   └── ...
└── ...
```

**关键概念：**
- **Thread（线程）**：最小执行单元
- **Block（线程块）**：一组线程，可以共享内存和同步
- **Grid（网格）**：所有线程块的集合

#### 2. 内存层次

| 内存类型 | 作用域 | 生命周期 | 访问速度 | 大小 |
|---------|--------|---------|----------|------|
| Register | Thread | Thread | 最快 | ~64KB |
| Shared Memory | Block | Block | 很快 | ~48KB |
| Local Memory | Thread | Thread | 慢 | 大 |
| Global Memory | Grid | Application | 慢 | GB级 |
| Constant Memory | Grid | Application | 快(缓存) | ~64KB |
| Texture Memory | Grid | Application | 快(缓存) | - |

---

## 第一个 CUDA Kernel

### 向量加法示例

#### C++ 主程序

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA Kernel 定义
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // 计算全局线程 ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1000000;
    size_t bytes = N * sizeof(float);

    // 1. 分配主机内存
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);

    // 2. 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. 拷贝数据到设备
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // 4. 配置并启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 5. 拷贝结果回主机
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. 验证结果
    for (int i = 0; i < 10; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    // 7. 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

### 关键点解析

#### 1. Kernel 函数定义
```cpp
__global__ void vectorAdd(...)
```
- `__global__`：在设备上执行，从主机调用
- `__device__`：在设备上执行，从设备调用
- `__host__`：在主机上执行（默认）

#### 2. 线程索引计算
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x`：当前块在网格中的索引
- `blockDim.x`：每个块的线程数
- `threadIdx.x`：当前线程在块中的索引

#### 3. Kernel 启动配置
```cpp
kernelName<<<blocksPerGrid, threadsPerBlock>>>(args);
```
- 第一个参数：网格维度（块的数量）
- 第二个参数：块维度（每个块的线程数）

---

## 内存层次结构

### 1. Global Memory（全局内存）

**特点：**
- 所有线程可访问
- 延迟高（400-800 cycles）
- 需要优化访问模式

**优化要点：**
```cpp
// ✅ 好的访问模式：Coalesced（合并访问）
__global__ void coalesced_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx;  // 连续访问
}

// ❌ 差的访问模式：Strided（跨步访问）
__global__ void strided_access(float* data, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx * stride] = idx;  // 跳跃访问
}
```

### 2. Shared Memory（共享内存）

**特点：**
- 块内线程共享
- 延迟低（~5 cycles）
- 容量小（48KB）

**使用示例：**
```cpp
__global__ void useSharedMemory(float* input, float* output, int N) {
    // 静态分配
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载到共享内存
    if (idx < N) {
        shared[tid] = input[idx];
    }
    __syncthreads();  // 同步等待所有线程加载完成

    // 使用共享内存进行计算
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared[i];
        }
        output[idx] = sum;
    }
}
```

**动态分配：**
```cpp
__global__ void dynamicShared(float* data) {
    extern __shared__ float shared[];  // 动态分配
    // ...
}

// 启动时指定大小
kernel<<<blocks, threads, sharedMemSize>>>(args);
```

### 3. Register（寄存器）

**特点：**
- 每个线程私有
- 访问最快
- 数量有限

**优化技巧：**
```cpp
// 复用寄存器变量
__global__ void registerOptimization(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 寄存器变量
    float temp1 = data[idx];
    float temp2 = temp1 * 2.0f;
    float temp3 = temp2 + 1.0f;

    data[idx] = temp3;
}
```

---

## 线程组织与索引

### 一维索引
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 二维索引（适用于矩阵）
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;
```

### 三维索引
```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### 最佳实践

```cpp
// 1. 选择合适的块大小（通常是 32 的倍数）
dim3 blockSize(16, 16);  // 256 threads per block
dim3 gridSize((width + 15) / 16, (height + 15) / 16);

// 2. 考虑 Warp 大小（32）
int threadsPerBlock = 256;  // 8 warps per block

// 3. 占用率优化
// 根据 SM 资源限制选择合适的块大小
```

---

## 完整示例：矩阵乘法

### Naive 实现

```cpp
__global__ void matrixMulNaive(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 使用 Shared Memory 优化

```cpp
#define TILE_SIZE 16

__global__ void matrixMulTiled(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 分块计算
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 加载 A 的 tile
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的 tile
        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算部分和
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## 性能优化要点

### 1. 内存访问优化
- 使用 Coalesced 访问模式
- 利用 Shared Memory 减少 Global Memory 访问
- 避免 Bank Conflicts

### 2. 计算优化
- 减少分支divergence
- 利用 ILP（指令级并行）
- 使用内置函数（`__fmaf`, `__syncthreads` 等）

### 3. 占用率优化
- 调整块大小以最大化占用率
- 平衡寄存器和共享内存使用
- 使用 `--ptxas-options=-v` 查看资源使用

### 4. 错误检查
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

CUDA_CHECK(cudaMalloc(&d_A, bytes));
```

---

## 编译与运行

### 编译命令
```bash
# 基本编译
nvcc -o vector_add vector_add.cu

# 指定架构
nvcc -arch=sm_80 -o vector_add vector_add.cu

# 优化编译
nvcc -O3 -use_fast_math -o vector_add vector_add.cu

# 查看资源使用
nvcc --ptxas-options=-v -o vector_add vector_add.cu
```

### 性能测试
```cpp
// 添加计时
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("Kernel time: %f ms\n", milliseconds);
```

---

## 下一步

查看 [examples](examples/) 目录中的完整代码示例：
- `vector_add.cu` - 向量加法
- `matrix_mul.cu` - 矩阵乘法
- `reduce.cu` - 规约操作
- `transpose.cu` - 矩阵转置

继续学习 [性能优化指南](../03_performance_optimization/optimization_guide.md)
