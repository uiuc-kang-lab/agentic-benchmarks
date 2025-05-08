#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// This kernel uses tiling with shared memory, ensuring that global memory accesses are coalesced
// by having threads in a warp load consecutive elements. 
__global__ void AlignedTiledMatmulKernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Compute the row and column index for this thread
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    // Number of tiles needed to cover the K dimension
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;  // Column to load from A
        int bRow = t * TILE_WIDTH + threadIdx.y;  // Row to load from B

        // Coalesced load for A: threads in the same row load consecutive elements
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // Coalesced load for B: threads in the same row (of the tile) load consecutive elements
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// PyTorch binding
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AlignedTiledMatmulKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aligned tiled matrix multiplication with coalesced accesses (CUDA)");
}
