#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Define block dimension (tile size)
#define BLOCK_DIM 16

// Kernel optimized to minimize __syncthreads() calls
__global__ void matmul_min_sync_kernel(const float* __restrict__ A, 
                                         const float* __restrict__ B, 
                                         float* __restrict__ C, 
                                         int M, int N, int K) {
    // Calculate global row and column indices
    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    float sum = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float sA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sB[BLOCK_DIM][BLOCK_DIM];

    // Number of tiles along the K dimension
    int numTiles = (K + BLOCK_DIM - 1) / BLOCK_DIM;

    // Load the first tile (tile 0) into shared memory
    int tiledIndex = t * BLOCK_DIM;
    if (row < M && (tiledIndex + threadIdx.x) < K)
        sA[threadIdx.y][threadIdx.x] = A[row * K + (tiledIndex + threadIdx.x)];
    else
        sA[threadIdx.y][threadIdx.x] = 0.0f;

    if ((tiledIndex + threadIdx.y) < K && col < N)
        sB[threadIdx.y][threadIdx.x] = B[(tiledIndex + threadIdx.y) * N + col];
    else
        sB[threadIdx.y][threadIdx.x] = 0.0f;

    // Loop over tiles, loading the next tile concurrently with computation
    for (int t = 0; t < numTiles - 1; t++) {
        __syncthreads(); // Ensure current tile is loaded before computation

        // Compute partial sum using the current tile
        #pragma unroll
        for (int k = 0; k < BLOCK_DIM; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Load next tile into shared memory, overwriting current tile
        int t_next = t + 1;
        int tiledIndex = t_next * BLOCK_DIM;
        if (row < M && (tiledIndex + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + (tiledIndex + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((tiledIndex + threadIdx.y) < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[(tiledIndex + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Synchronize to ensure the last tile is fully loaded
    __syncthreads();
    
    // Compute partial sum for the final tile
    #pragma unroll
    for (int k = 0; k < BLOCK_DIM; k++) {
        sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    }

    // Write the result back to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    matmul_min_sync_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication optimized with minimal __syncthreads() calls");
}
