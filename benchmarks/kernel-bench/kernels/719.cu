#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size
#define TILE_WIDTH 16

// Hybrid kernel: Tiled matrix multiplication with shared memory re-use and __ldg for read-only caching.
__global__ void HybridTiledLdgKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
    // Compute output row and column indices
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float cValue = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Number of tiles in the K dimension
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;

        // Use __ldg for read-only caching from global memory
        float aElem = (row < M && tiledCol < K) ? __ldg(&A[row * K + tiledCol]) : 0.0f;
        float bElem = (tiledRow < K && col < N) ? __ldg(&B[tiledRow * N + col]) : 0.0f;

        As[threadIdx.y][threadIdx.x] = aElem;
        Bs[threadIdx.y][threadIdx.x] = bElem;
        
        __syncthreads();

        // Multiply the two tiles together
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            cValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }
    
    // Write result if within bounds
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

// Host function: validates inputs, sets grid/block and launches kernel

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Use TILE_WIDTH x TILE_WIDTH block configuration
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    HybridTiledLdgKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(),
                                                B.data_ptr<float>(),
                                                C.data_ptr<float>(),
                                                M, K, N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid tiled matrix multiplication with __ldg caching (CUDA)");
}
