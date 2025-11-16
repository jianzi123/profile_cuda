/**
 * Vector Add v1: Coalesced Memory Access
 *
 * 优化: 修复内存合并访问
 * - Sequential memory access pattern
 * - Perfect memory coalescing
 * - Expected sectors_per_request: ~1.0
 *
 * 预期提升: 5-10x 相比 v0
 * 预期性能: 在 A100 上处理 256MB 数据约 1.5-2.0 ms
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ✅ Coalesced memory access kernel
__global__ void vector_add_coalesced(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c,
                                      int n) {
    // Coalesced access: adjacent threads access adjacent memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop for processing large arrays
    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
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

    size_t bytes = n * sizeof(float);
    printf("=== Vector Add v1: Coalesced Memory Access ===\n");
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

    // Launch configuration
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    printf("\nLaunch config: %d blocks x %d threads\n", blocks, threads_per_block);

    // Warmup
    vector_add_coalesced<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        vector_add_coalesced<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
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
    printf("Key changes from v0:\n");
    printf("  ✅ Sequential access: c[i] = a[i] + b[i]\n");
    printf("  ✅ No stride: adjacent threads access adjacent elements\n");
    printf("  ✅ Grid-stride loop for scalability\n");

    printf("\n=== NCU Analysis Hints ===\n");
    printf("Expected improvements:\n");
    printf("  - l1tex__average_t_sectors_per_request: ~1.0 (优化成功)\n");
    printf("  - Memory Throughput: 80-95%% (仍是显存瓶颈，正常)\n");
    printf("  - SM Throughput: 10-20%% (Element-wise 操作特征)\n");
    printf("  - Bandwidth: 接近理论峰值 1.5 TB/s\n");
    printf("\nNCU command:\n");
    printf("  ncu --set full --export v1_coalesced ./v1_coalesced\n");
    printf("  ncu --metrics l1tex__average_t_sectors_per_request ./v1_coalesced\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
