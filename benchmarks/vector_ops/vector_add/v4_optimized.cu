/**
 * Vector Add v4: Final Optimized Version
 *
 * 综合最佳实践:
 * - float4 vectorization for memory efficiency
 * - Optimized grid-stride loop
 * - Read-only cache hints (__ldg or __restrict__)
 * - Compile-time optimizations
 *
 * 预期性能: 在 A100 上达到 95%+ 带宽利用率
 * 处理 256MB 数据约 0.6-0.8 ms
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ✅ Final optimized kernel
__global__ void vector_add_optimized(const float4* __restrict__ a,
                                      const float4* __restrict__ b,
                                      float4* __restrict__ c,
                                      int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop with vectorization
    #pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        // Read-only cache optimization (automatic with __restrict__)
        float4 a_val = a[i];
        float4 b_val = b[i];

        float4 c_val;
        c_val.x = a_val.x + b_val.x;
        c_val.y = a_val.y + b_val.y;
        c_val.z = a_val.z + b_val.z;
        c_val.w = a_val.w + b_val.w;

        c[i] = c_val;
    }
}

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void init_array(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verify_result(const float* a, const float* b, const float* c, int n) {
    for (int i = 0; i < n; i++) {
        float expected = a[i] + b[i];
        if (fabs(c[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: expected %f, got %f\n",
                   i, expected, c[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Problem size: 64M elements = 256 MB per array
    int n = 64 * 1024 * 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    n = (n + 3) / 4 * 4;  // Round up to multiple of 4

    size_t bytes = n * sizeof(float);
    printf("=== Vector Add v4: Final Optimized ===\n");
    printf("Problem size: %d elements (%.2f MB per array)\n", n, bytes / 1024.0 / 1024.0);
    printf("Total memory: %.2f MB\n", 3 * bytes / 1024.0 / 1024.0);

    // Get device properties for optimal configuration
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Memory Bandwidth: %.1f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize arrays
    init_array(h_a, n);
    init_array(h_b, n);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    check_cuda_error(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda_error(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda_error(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    // Copy data to device
    check_cuda_error(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "H2D a");
    check_cuda_error(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "H2D b");

    // Optimal launch configuration
    int n_vec = n / 4;
    int threads_per_block = 256;  // Good balance for most GPUs
    // Use enough blocks to saturate GPU
    int num_sms = prop.multiProcessorCount;
    int blocks = min((n_vec + threads_per_block - 1) / threads_per_block, num_sms * 8);

    printf("\nLaunch config: %d blocks x %d threads\n", blocks, threads_per_block);
    printf("Total threads: %d (processing %d float4 elements)\n",
           blocks * threads_per_block, n_vec);

    // Warmup
    vector_add_optimized<<<blocks, threads_per_block>>>(
        (float4*)d_a, (float4*)d_b, (float4*)d_c, n_vec);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        vector_add_optimized<<<blocks, threads_per_block>>>(
            (float4*)d_a, (float4*)d_b, (float4*)d_c, n_vec);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iterations;

    // Copy result back
    check_cuda_error(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "D2H c");

    // Verify
    bool correct = verify_result(h_a, h_b, h_c, n);

    // Calculate bandwidth and efficiency
    float total_gb = 3.0f * bytes / 1e9;  // Read a, b, write c
    float bandwidth = total_gb / (avg_ms / 1000.0f);
    float theoretical_bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    float efficiency = (bandwidth / theoretical_bw) * 100.0f;

    printf("\n=== Performance ===\n");
    printf("Time: %.4f ms\n", avg_ms);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    printf("Theoretical: %.2f GB/s\n", theoretical_bw);
    printf("Efficiency: %.1f%%\n", efficiency);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    printf("\n=== Final Optimizations Applied ===\n");
    printf("✅ Memory coalescing: Sequential access pattern\n");
    printf("✅ Vectorization: float4 (4 elements per transaction)\n");
    printf("✅ Loop unrolling: #pragma unroll 4\n");
    printf("✅ Read-only hints: __restrict__ qualifier\n");
    printf("✅ Optimal grid size: %d blocks for %d SMs\n", blocks, num_sms);
    printf("✅ Compile flags: -O3 -use_fast_math\n");

    printf("\n=== Performance Comparison (Expected) ===\n");
    printf("v0 (Naive):           10-15 ms   (100-150 GB/s)  Baseline\n");
    printf("v1 (Coalesced):       1.5-2.0 ms (700-900 GB/s)  ~8x faster\n");
    printf("v2 (Vectorized):      0.8-1.2 ms (1000-1300 GB/s) ~2x faster\n");
    printf("v3 (Shared Mem):      2.0-3.0 ms (400-600 GB/s)  ❌ Slower!\n");
    printf("v4 (Optimized):       0.6-0.8 ms (1400-1500 GB/s) ~1.5x faster\n");
    printf("\nTotal speedup: v4 vs v0 = ~20x\n");

    printf("\n=== NCU Analysis Hints ===\n");
    printf("Expected metrics:\n");
    printf("  - l1tex__average_t_sectors_per_request: ~1.0\n");
    printf("  - Memory Throughput: 90-95%%\n");
    printf("  - SM Throughput: 10-15%% (memory-bound 正常)\n");
    printf("  - smsp__sass_inst_executed_op_global_ld: 最小化\n");
    printf("  - Achieved bandwidth: 1400-1500 GB/s on A100\n");
    printf("\nNCU command:\n");
    printf("  ncu --set full --export v4_optimized ./v4_optimized\n");
    printf("\n对比所有版本:\n");
    printf("  ncu --set full --export v0_naive ./v0_naive\n");
    printf("  ncu --set full --export v1_coalesced ./v1_coalesced\n");
    printf("  ncu --set full --export v2_vectorized ./v2_vectorized\n");
    printf("  ncu --set full --export v3_shared ./v3_shared_tiling\n");
    printf("  ncu --set full --export v4_optimized ./v4_optimized\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
