/**
 * GEMM v2: Shared Memory Tiling
 *
 * 核心优化: 使用 Shared Memory 缓存 tile，实现数据重用
 *
 * 优化原理:
 * - 将 A 和 B 分块加载到 Shared Memory
 * - 每个 tile 被重用 TILE_SIZE 次
 * - 减少 global memory 访问 TILE_SIZE 倍
 *
 * 预期提升: 10-15x vs v0
 * 预期性能: ~1500-3000 GFLOPS on A100
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32  // 32×32 tile (1KB shared memory per tile)

// ✅ Shared Memory Tiling kernel
__global__ void gemm_shared_tiling(
    const float* __restrict__ A,  // M × K
    const float* __restrict__ B,  // K × N
    float* __restrict__ C,         // M × N
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread position in block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global position
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for result
    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A (collaborative loading)
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;  // Padding
        }

        // Load tile from B (collaborative loading)
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;  // Padding
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product using shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
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

    printf("=== GEMM v2: Shared Memory Tiling ===\n");
    printf("Matrix sizes: A(%d×%d) × B(%d×%d) = C(%d×%d)\n", M, K, K, N, M, N);
    printf("Tile size: %d × %d\n", TILE_SIZE, TILE_SIZE);

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
    dim3 block(TILE_SIZE, TILE_SIZE);  // 1024 threads per block
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("\nLaunch config: Grid(%d, %d) Block(%d, %d)\n",
           grid.x, grid.y, block.x, block.y);

    size_t shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    printf("Shared memory per block: %zu bytes (%.2f KB)\n",
           shared_mem_size, shared_mem_size / 1024.0);

    // Warmup
    gemm_shared_tiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        gemm_shared_tiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    double gflops = (2.0 * M * N * K) / (avg_ms / 1000.0) / 1e9;

    printf("\n=== Performance ===\n");
    printf("Time: %.4f ms\n", avg_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Efficiency: %.2f%% (vs A100 FP32 peak 19500 GFLOPS)\n",
           (gflops / 19500.0) * 100);

    printf("\n=== Optimization Details ===\n");
    printf("数据重用分析:\n");
    printf("  - 每个 A tile 被读 1 次，重用 TILE_SIZE = %d 次\n", TILE_SIZE);
    printf("  - 每个 B tile 被读 1 次，重用 TILE_SIZE = %d 次\n", TILE_SIZE);
    printf("  - Global memory 访问减少约 %dx\n", TILE_SIZE);
    printf("\nShared Memory 使用:\n");
    printf("  - 每个 block: 2 × %d × %d × 4 bytes = %.2f KB\n",
           TILE_SIZE, TILE_SIZE, shared_mem_size / 1024.0);
    printf("  - A100 每个 SM: 192 KB (可同时运行多个 block)\n");
    printf("\nArithmetic Intensity 提升:\n");
    printf("  v0: 每读 1 float 做 1 FLOP → AI = 0.25 FLOPS/Byte\n");
    printf("  v2: 每读 1 float 做 %d FLOPS → AI = %.2f FLOPS/Byte\n",
           TILE_SIZE, TILE_SIZE / 4.0);
    printf("  → 仍然 Memory-bound, 但更接近 Ridge Point\n");

    printf("\n=== NCU Analysis Hints ===\n");
    printf("预期改进:\n");
    printf("  - dram__bytes.sum: 减少 ~%dx (数据重用)\n", TILE_SIZE);
    printf("  - SM Throughput: 提升到 20-30%% (更多计算)\n");
    printf("  - Memory Throughput: 仍然 70-85%% (仍是瓶颈但改善)\n");
    printf("  - Occupancy: 检查是否受 Shared Memory 限制\n");
    printf("\n潜在问题:\n");
    printf("  - Bank conflict: As[ty][k] 和 Bs[k][tx] 列访问\n");
    printf("  - 非向量化访问\n");
    printf("  → 参见 v3_optimized\n");
    printf("\nNCU command:\n");
    printf("  ncu --set full --export gemm_v2_shared ./gemm_v2_shared_tiling %d %d %d\n", M, N, K);
    printf("  ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_shared ./gemm_v2_shared_tiling\n");

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
