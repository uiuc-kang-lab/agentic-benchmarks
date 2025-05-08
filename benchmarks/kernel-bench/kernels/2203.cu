#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions and other constants
#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32
#define THRESHOLD_SIZE 512  // Threshold to switch between implementations

__global__ void tiledSharedLdgKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int K, int M, int N) {
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;
    
    float sum = 0.0f;
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;

    #pragma unroll 2
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Vectorized loads for better memory bandwidth
        for (int index = tid; index < BLOCK_K * TILE_M; index += totalThreads) {
            int t = index / TILE_M;
            int m = index % TILE_M;
            int global_k = k0 + t;
            int global_m = blockIdx.x * TILE_M + m;
            As[t][m] = (global_k < K && global_m < M) ? __ldg(&A[global_k * M + global_m]) : 0.0f;
        }

        for (int index = tid; index < BLOCK_K * TILE_N; index += totalThreads) {
            int t = index / TILE_N;
            int n = index % TILE_N;
            int global_k = k0 + t;
            int global_n = blockIdx.y * TILE_N + n;
            Bs[t][n] = (global_k < K && global_n < N) ? __ldg(&B[global_k * N + global_n]) : 0.0f;
        }

        __syncthreads();

        #pragma unroll 8
        for (int t = 0; t < BLOCK_K; t++) {
            sum += As[t][threadIdx.x] * Bs[t][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void linearNoDivKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int K, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = M * N;
    
    // Use vectorized loads for better memory throughput
    float4 a_vec, b_vec;
    
    for (; idx < total; idx += stride) {
        int i = idx / N;
        int j = idx % N;
        float sum = 0.0f;
        
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            sum += __ldg(&A[k * M + i]) * __ldg(&B[k * N + j]);
        }
        C[i * N + j] = sum;
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
    
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Choose implementation based on matrix size
    if (M > THRESHOLD_SIZE && N > THRESHOLD_SIZE) {
        dim3 block(TILE_M, TILE_N);
        dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
        tiledSharedLdgKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    } else {
        int threads = 256;
        int blocks = (M * N + threads - 1) / threads;
        linearNoDivKernel<<<blocks, threads>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid tiled/linear matrix multiplication (CUDA)");
}