"""
Matrix Multiplication Algorithms in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_cpu
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: C[m*N + n] = sum over k of A[m*K + k] * B[k*N + n]

    for (int row=0; row < M; row++) {
        for (int col=0; col<N; col++) {
            C[row * N + col] = 0;
            for (int k=0; k<K; k++) {
                C[row * N + col] += A[row * K + k] * B[k * N + col];
            }
        }
    }
}

# Step 2 - max_abs_diff
#include <cmath>

float max_abs_diff(const float* a, const float* b, int n) {
    // TODO: largest |a[i] - b[i]| over n elements, 0 for n == 0
    float max = 0;
    for (int i=0; i<n; i++) {
        float diff = fabsf(a[i]-b[i]);
        if (diff > max) max = diff;
    }
    return max;
}

# Step 3 - matmul_naive_kernel
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    C[row * N + col] = 0;
    for (int i=0; i<K; i++) {
        C[row * N + col] += A[row * K + i] * B[i * N + col];
    }
    return;
    
}

void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: 16x16 blocks, grid ((M+15)/16, (N+15)/16)
    
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * N * K);
    cudaMalloc(&d_C, sizeof(float) * M * N);
    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M+16-1)/16, (N+16-1)/16);

    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C, d_C, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

}

# Step 4 - matmul_coalesced_kernel
#include <cuda_runtime.h>

__global__ void matmul_coalesced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    C[row * N + col] = 0;
    for (int i=0; i<K; i++) {
        C[row * N + col] += A[row * K + i] * B[i * N + col];
    }
    return;
}

void launch_matmul_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * N * K);
    cudaMalloc(&d_C, sizeof(float) * M * N);
    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N+16-1)/16, (M+16-1)/16);

    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C, d_C, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
}

# Step 5 - time_launch_ms
#include <cuda_runtime.h>

typedef void (*matmul_launch_fn)(const float*, const float*, float*, int, int, int);

float time_launch_ms(matmul_launch_fn launch, const float* d_A, const float* d_B, float* d_C, int M, int N, int K, int iters) {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // dim3 block(16, 16);
    // dim3 grid((N+16-1)/16, (M+16-1)/16);

    launch(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    float total_time = 0.0;
    float iter_time = 0.0;

    for (int i=0; i<iters; i++) {
        cudaEventRecord(start);
        launch(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iter_time, start, stop);
        total_time += iter_time;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_time/iters;

}

double matmul_gflops(int M, int N, int K, float ms) {
    double total_flops = (2.0 * M * N * K);
    double gflops = total_flops /(ms * 1e-3)/ 1e9;
    return gflops;

}

# Step 6 - matmul_tiled_kernel
#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    //__shared__ As[TILE][TILE], Bs[TILE][TILE]; loop over K tiles: guarded loads, sync, 16 FMAs, sync
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t row = size_t(blockIdx.y) * TILE + ty;
    const size_t col = size_t(blockIdx.x) * TILE + tx;
    float sum = 0.0f;

    for (size_t k0 = 0; k0 < K; k0 += TILE) {
        const size_t a_col = k0 + tx;
        const size_t b_row = k0 + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();  // Both tiles must be ready before use.

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // Finish reading before anyone overwrites the tiles.
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * N * K);
    cudaMalloc(&d_C, sizeof(float) * M * N);
    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N * K, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N+16-1)/16, (M+16-1)/16);

    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(C, d_C, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
}

# Step 7 - matmul_tiled_1d_kernel (not yet solved)
# TODO: implement

# Step 8 - matmul_tiled_2d_kernel (not yet solved)
# TODO: implement

# Step 9 - matmul_vectorized_kernel (not yet solved)
# TODO: implement

# Step 10 - matmul_double_buffered_kernel (not yet solved)
# TODO: implement

# Step 11 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 12 - matmul_batched_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_splitk_kernel (not yet solved)
# TODO: implement

# Step 14 - gemv_kernel (not yet solved)
# TODO: implement

# Step 15 - matmul_bias_relu_kernel (not yet solved)
# TODO: implement

# Step 16 - matrix_addsub_kernel (not yet solved)
# TODO: implement

# Step 17 - strassen_one_level (not yet solved)
# TODO: implement

# Step 18 - csr_spmm_kernel (not yet solved)
# TODO: implement

# Step 19 - matmul_lower_triangular_kernel (not yet solved)
# TODO: implement

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

