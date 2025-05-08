#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile size for shared memory tiling
#define TILE_SIZE 16

// Device function to load tiles of A and B into shared memory
__device__ void loadTiles(const float* __restrict__ A, const float* __restrict__ B,
                          float tileA[TILE_SIZE][TILE_SIZE], float tileB[TILE_SIZE][TILE_SIZE],
                          int K, int M, int N, int t, int row, int col) {
    int aIndex = t * TILE_SIZE + threadIdx.x;
    if (row < M && aIndex < K) {
        tileA[threadIdx.y][threadIdx.x] = A[aIndex * M + row];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    int bIndex = t * TILE_SIZE + threadIdx.y;
    if (bIndex < K && col < N) {
        tileB[threadIdx.y][threadIdx.x] = B[bIndex * N + col];
    } else {
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to compute the dot product for a tile
__device__ float computeTileDotProduct(float tileA[TILE_SIZE][TILE_SIZE], float tileB[TILE_SIZE][TILE_SIZE]) {
    float sum = 0.0f;
    #pragma unroll
    for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
        sum += tileA[threadIdx.y][k_inner] * tileB[k_inner][threadIdx.x];
    }
    return sum;
}

// CUDA kernel for computing C = A.T * B using shared memory tiling.
__global__ void matMulSharedKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K, int M, int N) {
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        loadTiles(A, B, tileA, tileB, K, M, N, t, row, col);
        __syncthreads();

        sum += computeTileDotProduct(tileA, tileB);
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

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    matMulSharedKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using shared memory tiling");
}
