/**
 * Vector Add v3: Shared Memory Tiling (Over-optimization Example)
 *
 * 警告: 这是一个反面教材 - 错误地使用 Shared Memory
 * - Vector add 不需要数据重用
 * - Shared Memory 引入额外开销
 * - 性能反而下降
 *
 * 目的: 演示何时 NOT 应该使用某种优化技术
 * 预期性能: 比 v2 慢 2-3x (由于额外的 sync 和 shared memory overhead)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 256

// ❌ Incorrect use of Shared Memory for vector add
__global__ void vector_add_shared_tiling(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ c,
                                          int n) {
    __shared__ float s_a[TILE_SIZE];
    __shared__ float s_b[TILE_SIZE];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * TILE_SIZE + tid;

    // Load to shared memory (unnecessary overhead)
    if (global_idx < n) {
        s_a[tid] = a[global_idx];
        s_b[tid] = b[global_idx];
    }
    __syncthreads();  // Synchronization overhead

    // Compute (no data reuse, shared memory wasted)
    if (global_idx < n) {
        c[global_idx] = s_a[tid] + s_b[tid];
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
    printf("=== Vector Add v3: Shared Memory Tiling (反面教材) ===\n");
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
    int threads_per_block = TILE_SIZE;
    int blocks = (n + TILE_SIZE - 1) / TILE_SIZE;

    printf("\nLaunch config: %d blocks x %d threads\n", blocks, threads_per_block);
    printf("Shared memory per block: %zu bytes\n", 2 * TILE_SIZE * sizeof(float));

    // Warmup
    vector_add_shared_tiling<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        vector_add_shared_tiling<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
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

    printf("\n=== 为什么这是错误的优化? ===\n");
    printf("❌ 问题 1: 没有数据重用\n");
    printf("   - Vector add 每个元素只读一次，写一次\n");
    printf("   - Shared memory 的优势在于数据重用 (如 GEMM, convolution)\n");
    printf("\n❌ 问题 2: 引入额外开销\n");
    printf("   - __syncthreads() 导致 warp stall\n");
    printf("   - Global → Shared → Register 多一次搬运\n");
    printf("\n❌ 问题 3: 性能反而下降\n");
    printf("   - 预期比 v2 慢 2-3x\n");
    printf("   - NCU 会显示 barrier stall 增加\n");

    printf("\n=== NCU Analysis Hints ===\n");
    printf("预期问题:\n");
    printf("  - smsp__average_warps_issue_stalled_barrier: 20-40%% (barrier 等待)\n");
    printf("  - 更多的指令数 (多了 shared memory load/store)\n");
    printf("  - 更低的 bandwidth (由于 overhead)\n");
    printf("\n何时应该使用 Shared Memory:\n");
    printf("  ✅ Matrix multiply (每个元素读 K 次)\n");
    printf("  ✅ Convolution (kernel 重用)\n");
    printf("  ✅ Histogram (原子操作优化)\n");
    printf("  ❌ Element-wise operations (无重用)\n");
    printf("\nNCU command:\n");
    printf("  ncu --set full --export v3_shared_tiling ./v3_shared_tiling\n");
    printf("  ncu --metrics smsp__average_warps_issue_stalled_barrier ./v3_shared_tiling\n");

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
