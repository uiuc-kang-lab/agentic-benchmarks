#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile dimension and split factor for the K dimension
#define TILE_DIM 16
#define SPLIT_K 2

// Optimized CUDA kernel for computing C = A.T * B.
// This combines shared memory tiling and split-K parallelism.
__global__ void optimizedMatmulKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int K, int M, int N) {
    int block_k_size = (K + SPLIT_K - 1) / SPLIT_K; // Calculate the size of each block for K dimension
    int k_start = blockIdx.z * block_k_size;
    int k_end = min(k_start + block_k_size, K);

    int row = blockIdx.x * TILE_DIM + threadIdx.y;
    int col = blockIdx.y * TILE_DIM + threadIdx.x;

    float cValue = 0.0f;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int local_k = k_end - k_start;
    int numTiles = (local_k + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; t++) {
        int k_idx = t * TILE_DIM + threadIdx.x;
        int global_k = k_start + k_idx;
        As[threadIdx.y][threadIdx.x] = (row < M && global_k < k_end) ? A[global_k * M + row] : 0.0f;

        int k_idx_b = t * TILE_DIM + threadIdx.y;
        global_k = k_start + k_idx_b;
        Bs[threadIdx.y][threadIdx.x] = (col < N && global_k < k_end) ? B[global_k * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            cValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], cValue);
    }
}

// The forward function exposed via PyBind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((M + TILE_DIM - 1) / TILE_DIM,
                 (N + TILE_DIM - 1) / TILE_DIM,
                 SPLIT_K);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    optimizedMatmulKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B optimized with shared memory and split-K parallelism (CUDA)");
}
