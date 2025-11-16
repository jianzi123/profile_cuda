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

// ============================================================================
// 模式 1: 标准一对一映射
// ============================================================================
__global__ void pattern1_standard(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// 模式 2: Grid-Stride Loop
// ============================================================================
__global__ void pattern2_grid_stride(float* out, const float* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 每个线程处理多个元素
    for (int i = tid; i < N; i += stride) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
}

// ============================================================================
// 模式 3: 向量化 (float4)
// ============================================================================
__global__ void pattern3_vectorized(float* out, const float* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 4;

    if (base + 3 < N) {
        float4 val = reinterpret_cast<const float4*>(in)[idx];

        val.x = val.x * 2.0f + 1.0f;
        val.y = val.y * 2.0f + 1.0f;
        val.z = val.z * 2.0f + 1.0f;
        val.w = val.w * 2.0f + 1.0f;

        reinterpret_cast<float4*>(out)[idx] = val;
    }

    // 处理剩余元素
    for (int i = base; i < min(base + 4, N); i++) {
        if (i >= (base & ~3) + 4) {
            out[i] = in[i] * 2.0f + 1.0f;
        }
    }
}

// ============================================================================
// 模式 4: 向量化 + Grid-Stride (最优化)
// ============================================================================
__global__ void pattern4_vectorized_grid_stride(float* out, const float* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 向量化处理
    int vec_n = N / 4;
    for (int i = tid; i < vec_n; i += stride) {
        float4 val = reinterpret_cast<const float4*>(in)[i];

        val.x = val.x * 2.0f + 1.0f;
        val.y = val.y * 2.0f + 1.0f;
        val.z = val.z * 2.0f + 1.0f;
        val.w = val.w * 2.0f + 1.0f;

        reinterpret_cast<float4*>(out)[i] = val;
    }

    // 处理剩余元素（每个线程处理一个）
    int remainder_start = vec_n * 4;
    for (int i = remainder_start + tid; i < N; i += stride) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
}

// ============================================================================
// ❌ 错误模式: 以 32 为步长（演示常见错误）
// ============================================================================
__global__ void pattern_wrong_stride32(float* out, const float* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ❌ 错误：这会导致某些线程重复处理相同数据
    for (int i = tid; i < N; i += 32) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
    // 问题：如果 tid = 0, 处理 0, 32, 64, ...
    //      如果 tid = 32, 也处理 32, 64, 96, ...
    //      导致竞争条件！
}

// ============================================================================
// Benchmark 函数
// ============================================================================
template<typename Kernel>
float benchmark(Kernel kernel, dim3 grid, dim3 block,
                float* d_out, const float* d_in, int N,
                const char* name, int iterations = 100) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block>>>(d_out, d_in, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_out, d_in, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= iterations;

    // 计算带宽
    float bytes = 2.0f * N * sizeof(float);  // Read + Write
    float bandwidth = (bytes / 1e9) / (milliseconds / 1000);

    printf("%-35s: %6.3f ms, %7.2f GB/s", name, milliseconds, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return milliseconds;
}

// ============================================================================
// 验证结果
// ============================================================================
bool verify(const std::vector<float>& result, const std::vector<float>& input) {
    for (size_t i = 0; i < result.size(); i++) {
        float expected = input[i] * 2.0f + 1.0f;
        if (std::abs(result[i] - expected) > 1e-5) {
            std::cerr << "Error at index " << i << ": "
                      << "expected " << expected << ", got " << result[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    int N = 100000000;  // 100M elements

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "CUDA Thread Pattern Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Array size: " << N << " elements ("
              << (N * sizeof(float) / 1e6) << " MB)" << std::endl;
    std::cout << std::endl;

    // 分配内存
    size_t bytes = N * sizeof(float);
    std::vector<float> h_in(N);
    std::vector<float> h_out(N);

    // 初始化输入
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i % 1000) / 1000.0f;
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // 配置不同的启动参数
    int blockSize = 256;

    std::cout << "Testing different thread patterns:" << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // 模式 1: 标准映射
    // ========================================================================
    {
        int gridSize = (N + blockSize - 1) / blockSize;
        dim3 grid(gridSize);
        dim3 block(blockSize);

        float time = benchmark(pattern1_standard, grid, block, d_out, d_in, N,
                              "Pattern 1: Standard mapping");

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        if (verify(h_out, h_in)) {
            std::cout << "  ✓ Correct" << std::endl;
        } else {
            std::cout << "  ✗ Failed" << std::endl;
        }
    }

    // ========================================================================
    // 模式 2: Grid-Stride Loop
    // ========================================================================
    {
        int gridSize = 1024;  // 固定数量的 blocks
        dim3 grid(gridSize);
        dim3 block(blockSize);

        float time = benchmark(pattern2_grid_stride, grid, block, d_out, d_in, N,
                              "Pattern 2: Grid-stride loop");

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        if (verify(h_out, h_in)) {
            std::cout << "  ✓ Correct" << std::endl;
        } else {
            std::cout << "  ✗ Failed" << std::endl;
        }
    }

    // ========================================================================
    // 模式 3: 向量化 (float4)
    // ========================================================================
    {
        int gridSize = ((N / 4) + blockSize - 1) / blockSize;
        dim3 grid(gridSize);
        dim3 block(blockSize);

        float time = benchmark(pattern3_vectorized, grid, block, d_out, d_in, N,
                              "Pattern 3: Vectorized (float4)");

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        if (verify(h_out, h_in)) {
            std::cout << "  ✓ Correct" << std::endl;
        } else {
            std::cout << "  ✗ Failed" << std::endl;
        }
    }

    // ========================================================================
    // 模式 4: 向量化 + Grid-Stride (最优)
    // ========================================================================
    {
        int gridSize = 1024;
        dim3 grid(gridSize);
        dim3 block(blockSize);

        float time = benchmark(pattern4_vectorized_grid_stride, grid, block,
                              d_out, d_in, N,
                              "Pattern 4: Vectorized + Grid-stride");

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        if (verify(h_out, h_in)) {
            std::cout << "  ✓ Correct" << std::endl;
        } else {
            std::cout << "  ✗ Failed" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Testing different block sizes (Pattern 1):" << std::endl;
    std::cout << "========================================" << std::endl;

    // 测试不同的 block size
    int blockSizes[] = {64, 128, 256, 512, 1024};
    for (int bs : blockSizes) {
        int gridSize = (N + bs - 1) / bs;
        dim3 grid(gridSize);
        dim3 block(bs);

        char name[100];
        sprintf(name, "Block size: %4d", bs);
        benchmark(pattern1_standard, grid, block, d_out, d_in, N, name);
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Recommendations:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "1. Use block size of 256 (8 warps) for most cases" << std::endl;
    std::cout << "2. Use vectorization (float4) for memory-bound kernels" << std::endl;
    std::cout << "3. Use Grid-stride for very large data" << std::endl;
    std::cout << "4. Combine vectorization + Grid-stride for best performance" << std::endl;
    std::cout << std::endl;
    std::cout << "❌ DO NOT use stride=32 in loops!" << std::endl;
    std::cout << "✅ Use stride=blockDim.x*gridDim.x for Grid-stride" << std::endl;
    std::cout << std::endl;

    // 清理
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
