#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32
#define UNROLL_FACTOR 8

__global__ void unrolledTiledMatMulKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int K, int M, int N) {
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;
    
    float sum[2] = {0.0f, 0.0f};  // Maintain two partial sums for better instruction interleaving
    
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;

    // Manual unroll of outer loop for better instruction scheduling
    #pragma unroll 2
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Unrolled shared memory loads
        #pragma unroll 4
        for (int i = tid; i < BLOCK_K * TILE_M; i += totalThreads) {
            int t = i / TILE_M;
            int m = i % TILE_M;
            int global_k = k0 + t;
            int global_m = blockIdx.x * TILE_M + m;
            As[t][m] = (global_k < K && global_m < M) ? __ldg(&A[global_k * M + global_m]) : 0.0f;
        }

        #pragma unroll 4
        for (int i = tid; i < BLOCK_K * TILE_N; i += totalThreads) {
            int t = i / TILE_N;
            int n = i % TILE_N;
            int global_k = k0 + t;
            int global_n = blockIdx.y * TILE_N + n;
            Bs[t][n] = (global_k < K && global_n < N) ? __ldg(&B[global_k * N + global_n]) : 0.0f;
        }

        __syncthreads();

        // Manual unroll of computation loop
        #pragma unroll
        for (int t = 0; t < BLOCK_K; t += UNROLL_FACTOR) {
            // First set of unrolled operations
            sum[0] += As[t][threadIdx.x] * Bs[t][threadIdx.y];
            sum[1] += As[t+1][threadIdx.x] * Bs[t+1][threadIdx.y];
            
            #pragma unroll
            for (int u = 2; u < UNROLL_FACTOR; u += 2) {
                sum[0] += As[t+u][threadIdx.x] * Bs[t+u][threadIdx.y];
                sum[1] += As[t+u+1][threadIdx.x] * Bs[t+u+1][threadIdx.y];
            }
        }

        __syncthreads();
    }

    // Combine partial sums and write result
    if (row < M && col < N) {
        C[row * N + col] = sum[0] + sum[1];
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    unrolledTiledMatMulKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled tiled matrix multiplication (CUDA)");
}