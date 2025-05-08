#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 32

// Device function to load A tile in transposed format
__device__ void loadTileA(float* tileA, const float* A, int row, int tileIdx, int M, int K) {
    int aIndex = tileIdx * TILE_SIZE + threadIdx.x;
    if (row < M && aIndex < K) {
        tileA[threadIdx.y * (TILE_SIZE + 1) + threadIdx.x] = A[aIndex * M + row];
    } else {
        tileA[threadIdx.y * (TILE_SIZE + 1) + threadIdx.x] = 0.0f;
    }
}

// Device function to load B tile
__device__ void loadTileB(float* tileB, const float* B, int col, int tileIdx, int K, int N) {
    int bIndex = tileIdx * TILE_SIZE + threadIdx.y;
    if (bIndex < K && col < N) {
        tileB[threadIdx.y * (TILE_SIZE + 1) + threadIdx.x] = B[bIndex * N + col];
    } else {
        tileB[threadIdx.y * (TILE_SIZE + 1) + threadIdx.x] = 0.0f;
    }
}

// Device function to compute tile multiplication
__device__ float computeTileProduct(const float* tileA, const float* tileB) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        sum += tileA[threadIdx.y * (TILE_SIZE + 1) + k] * 
               tileB[k * (TILE_SIZE + 1) + threadIdx.x];
    }
    return sum;
}

__global__ void matMulModularKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K, int M, int N) {
    // Shared memory declaration with padding to avoid bank conflicts
    __shared__ float tileA[TILE_SIZE * (TILE_SIZE + 1)];
    __shared__ float tileB[TILE_SIZE * (TILE_SIZE + 1)];
    
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles collaboratively
        loadTileA(tileA, A, row, t, M, K);
        loadTileB(tileB, B, col, t, K, N);
        
        __syncthreads();
        
        // Compute partial product for this tile
        sum += computeTileProduct(tileA, tileB);
        
        __syncthreads();
    }

    // Store result if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    matMulModularKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using modular shared memory tiling");
}