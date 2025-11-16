#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA Kernel: Vector Addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 优化版本：使用向量化访问
__global__ void vectorAddOptimized(const float* A, const float* B, float* C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < N) {
        float4 a = reinterpret_cast<const float4*>(A)[idx / 4];
        float4 b = reinterpret_cast<const float4*>(B)[idx / 4];

        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        reinterpret_cast<float4*>(C)[idx / 4] = c;
    }
}

void runVectorAdd(int N) {
    std::cout << "Vector size: " << N << std::endl;

    size_t bytes = N * sizeof(float);

    // 1. 分配主机内存
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 2. 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 3. 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // 4. 配置并启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 创建 events 用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark naive version
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naive_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    naive_time /= 100;

    // Benchmark optimized version
    int optimizedBlocksPerGrid = ((N / 4) + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        vectorAddOptimized<<<optimizedBlocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float optimized_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&optimized_time, start, stop));
    optimized_time /= 100;

    // 5. 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // 6. 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (std::abs(h_C[i] - expected) > 1e-5) {
            std::cerr << "Error at index " << i << ": "
                      << "expected " << expected << ", got " << h_C[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "✓ Results verified!" << std::endl;
    }

    // 计算带宽
    float bytes_accessed = 3.0f * bytes;  // Read A, Read B, Write C
    float naive_bandwidth = (bytes_accessed / 1e9) / (naive_time / 1000);
    float optimized_bandwidth = (bytes_accessed / 1e9) / (optimized_time / 1000);

    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "Naive version:     " << naive_time << " ms, "
              << naive_bandwidth << " GB/s" << std::endl;
    std::cout << "Optimized version: " << optimized_time << " ms, "
              << optimized_bandwidth << " GB/s" << std::endl;
    std::cout << "Speedup:           " << (naive_time / optimized_time) << "x" << std::endl;

    // 7. 释放内存
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char** argv) {
    int N = 10000000;  // 10M elements

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    runVectorAdd(N);

    return 0;
}
