#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Tiled matrix multiplication kernel with branchless predicated loads to minimize warp divergence.
__global__ void MatmulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    // Compute global row and column indices for C
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float cValue = 0.0f;

    // Allocate shared memory for tiles of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Calculate the number of tiles needed in the K-dimension
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // Compute the column index for A and row index for B in the current tile
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;

        // Load elements from global memory into shared memory using predicated load
        // Ternary operator is used to avoid divergent branching inside the loop
        float aElem = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        float bElem = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;

        As[threadIdx.y][threadIdx.x] = aElem;
        Bs[threadIdx.y][threadIdx.x] = bElem;

        __syncthreads();

        // Multiply the two tiles together; the inner loop is unrolled for efficiency
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            cValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }
    
    // Write the final result to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

// The forward function checks input validity, allocates the output tensor, and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions based on tile size
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the kernel
    MatmulKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication (CUDA) with minimized warp divergence");
}
