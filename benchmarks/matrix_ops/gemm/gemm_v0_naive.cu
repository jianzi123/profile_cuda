/**
 * GEMM v0: Naive Implementation (Baseline)
 *
 * C = A × B
 * A: M × K
 * B: K × N
 * C: M × N
 *
 * 问题:
 * - 没有数据重用 (每次从 global memory 读取)
 * - 非合并访问 B 矩阵
 * - 大量重复访问显存
 *
 * 预期性能: ~100-200 GFLOPS on A100 (理论峰值 19.5 TFLOPS FP32)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ❌ Naive GEMM kernel
__global__ void gemm_naive(
    const float* __restrict__ A,  // M × K
    const float* __restrict__ B,  // K × N
    float* __restrict__ C,         // M × N
    int M, int N, int K
) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Compute C[row][col] = sum(A[row][k] * B[k][col])
        for (int k = 0; k < K; k++) {
            // ❌ 问题 1: A 每个元素读 N 次 (重复读取)
            // ❌ 问题 2: B 列访问非合并 (stride = N)
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

bool verify_result(const float* C, const float* C_ref, int M, int N, float eps = 1e-3) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_ref[i]) > eps) {
            printf("Verification failed at index %d: expected %f, got %f\n",
                   i, C_ref[i], C[i]);
            return false;
        }
    }
    return true;
}

// CPU reference implementation
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    // Problem size
    int M = 1024;
    int N = 1024;
    int K = 1024;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("=== GEMM v0: Naive Implementation ===\n");
    printf("Matrix sizes: A(%d×%d) × B(%d×%d) = C(%d×%d)\n", M, K, K, N, M, N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    printf("Memory: A=%.2f MB, B=%.2f MB, C=%.2f MB\n",
           size_A/1e6, size_B/1e6, size_C/1e6);

    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    // Initialize matrices
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    check_cuda_error(cudaMalloc(&d_A, size_A), "cudaMalloc A");
    check_cuda_error(cudaMalloc(&d_B, size_B), "cudaMalloc B");
    check_cuda_error(cudaMalloc(&d_C, size_C), "cudaMalloc C");

    // Copy to device
    check_cuda_error(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice), "H2D A");
    check_cuda_error(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice), "H2D B");

    // Launch configuration
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    printf("\nLaunch config: Grid(%d, %d) Block(%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    // Warmup
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iterations;

    // Copy result back
    check_cuda_error(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost), "D2H C");

    // Verify (small test)
    if (M <= 512 && N <= 512 && K <= 512) {
        printf("\nVerifying result...\n");
        gemm_cpu(h_A, h_B, h_C_ref, M, N, K);
        bool correct = verify_result(h_C, h_C_ref, M, N);
        printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    } else {
        printf("\nSkipping verification (matrix too large)\n");
    }

    // Calculate GFLOPS
    // GEMM: 2 * M * N * K operations (multiply-add)
    double gflops = (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e9;

    printf("\n=== Performance ===\n");
    printf("Time: %.4f ms\n", avg_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Efficiency: %.2f%% (vs A100 FP32 peak 19500 GFLOPS)\n",
           (gflops / 19500.0) * 100);

    printf("\n=== Problem Analysis ===\n");
    printf("每个 C 元素:\n");
    printf("  - 计算: 2K = %d FLOPS (1 次 FMA 算 2 FLOPS)\n", 2 * K);
    printf("  - 读取: A 读 K 次 + B 读 K 次 = %d reads\n", 2 * K);
    printf("  - Arithmetic Intensity: 2K / (2K × 4 bytes) = 0.25 FLOPS/Byte\n");
    printf("  → Memory-bound (AI << Ridge Point ~12.5)\n");
    printf("\n全局访问特征:\n");
    printf("  - A 每个元素被读 N = %d 次 (重复!)\n", N);
    printf("  - B 每个元素被读 M = %d 次 (重复!)\n", M);
    printf("  - B 列访问 stride = N (非合并!)\n");
    printf("\n优化方向:\n");
    printf("  ✓ 使用 Shared Memory 缓存 tile → 减少重复读取\n");
    printf("  ✓ 转置 B 或调整访问模式 → 提升合并\n");

    printf("\n=== NCU Analysis Hints ===\n");
    printf("预期问题:\n");
    printf("  - Memory Throughput: 80-95%% (显存瓶颈)\n");
    printf("  - SM Throughput: 5-10%% (计算单元闲置)\n");
    printf("  - l1tex__average_t_sectors_per_request: > 1.5 (B 矩阵列访问)\n");
    printf("  - dram__bytes.sum: 非常高 (大量重复读取)\n");
    printf("\nNCU command:\n");
    printf("  ncu --set full --export gemm_v0_naive ./gemm_v0_naive %d %d %d\n", M, N, K);
    printf("  ncu --metrics l1tex__average_t_sectors_per_request,dram__bytes.sum ./gemm_v0_naive\n");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
