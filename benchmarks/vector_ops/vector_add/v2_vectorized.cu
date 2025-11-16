/**
 * Vector Add v2: Vectorized Memory Access (float4)
 *
 * 优化: 向量化访问减少指令数
 * - Use float4 for 128-bit loads/stores
 * - Reduce number of memory transactions
 * - Better instruction-level parallelism
 *
 * 预期提升: 1.5-2x 相比 v1
 * 预期性能: 在 A100 上处理 256MB 数据约 0.8-1.2 ms
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ✅ Vectorized kernel using float4
__global__ void vector_add_vectorized(const float4* __restrict__ a,
                                       const float4* __restrict__ b,
                                       float4* __restrict__ c,
                                       int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Each thread processes 4 floats at once
    for (int i = tid; i < n; i += stride) {
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
    // Must be multiple of 4 for float4
    int n = 64 * 1024 * 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    n = (n + 3) / 4 * 4;  // Round up to multiple of 4

    size_t bytes = n * sizeof(float);
    printf("=== Vector Add v2: Vectorized (float4) ===\n");
    printf("Problem size: %d elements (%.2f MB per array)\n", n, bytes / 1024.0 / 1024.0);
    printf("Total memory: %.2f MB\n", 3 * bytes / 1024.0 / 1024.0);

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

    // Launch configuration - fewer threads needed since each processes 4 elements
    int n_vec = n / 4;  // Number of float4 elements
    int threads_per_block = 256;
    int blocks = (n_vec + threads_per_block - 1) / threads_per_block;

    printf("\nLaunch config: %d blocks x %d threads (processing %d float4 elements)\n",
           blocks, threads_per_block, n_vec);

    // Warmup
    vector_add_vectorized<<<blocks, threads_per_block>>>(
        (float4*)d_a, (float4*)d_b, (float4*)d_c, n_vec);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        vector_add_vectorized<<<blocks, threads_per_block>>>(
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

    // Calculate bandwidth
    float total_gb = 3.0f * bytes / 1e9;  // Read a, b, write c
    float bandwidth = total_gb / (avg_ms / 1000.0f);

    printf("\n=== Performance ===\n");
    printf("Time: %.4f ms\n", avg_ms);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    printf("\n=== Optimization Details ===\n");
    printf("Key changes from v1:\n");
    printf("  ✅ float4 vectorization: 4 elements per transaction\n");
    printf("  ✅ Reduced instruction count: 4x fewer loads/stores\n");
    printf("  ✅ Better ILP: compiler can schedule operations\n");
    printf("  ✅ Fewer threads needed: %d vs %d\n", n_vec, n);

    printf("\n=== NCU Analysis Hints ===\n");
    printf("Expected improvements:\n");
    printf("  - Instruction count: 减少约 50%%\n");
    printf("  - l1tex__t_bytes_per_sector_mem_global_op_ld: 16 bytes (float4)\n");
    printf("  - 更好的 pipeline 效率\n");
    printf("  - Bandwidth: 接近或达到理论峰值\n");
    printf("\nNCU command:\n");
    printf("  ncu --set full --export v2_vectorized ./v2_vectorized\n");
    printf("  ncu --metrics smsp__sass_inst_executed_op_global_ld ./v2_vectorized\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
