#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 32

// Declare constant memory for matrices (assuming they fit in constant memory limits for demonstration)
__constant__ float const_A[1024*1024];  // Example size, adjust based on actual use-case and constraints
__constant__ float const_B[1024*1024];  // Example size, adjust based on actual use-case and constraints

// CUDA kernel for computing C = A.T * B using optimized shared memory synchronization.
__global__ void matMulConstOptimizedKernel(float* __restrict__ C,
                                           int K, int M, int N) {
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aIndex = t * TILE_SIZE + threadIdx.x;
        if (row < M && aIndex < K)
            tileA[threadIdx.y][threadIdx.x] = const_A[aIndex * M + row];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        int bIndex = t * TILE_SIZE + threadIdx.y;
        if (bIndex < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = const_B[bIndex * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
            sum += tileA[threadIdx.y][k_inner] * tileB[k_inner][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Copy A and B to constant memory
    cudaMemcpyToSymbol(const_A, A.data_ptr<float>(), A.numel() * sizeof(float));
    cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), B.numel() * sizeof(float));

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    float* C_ptr = C.data_ptr<float>();

    matMulConstOptimizedKernel<<<gridDim, blockDim>>>(C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using constant memory optimization");
}
