#include <torch/extension.h>
#include <cuda_runtime.h>

// This kernel is optimized for a small K dimension and ensures memory coalescing by using
// vectorized loads for matrix B. We require that the number of columns in B (and C) is divisible by 4.

#define TILE_SIZE 32

// Kernel using shared memory tiling with vectorized load (float4) for matrix B
__global__ void matmul_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int K, int N) {
    // Shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    // Row and column index of the C element computed by this thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    // Number of tiles needed to cover the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A (scalar load, already coalesced since A is row-major)
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && A_col < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B using vectorized loads via float4 to improve coalescing
        // Only a subset of threads (first TILE_SIZE/4 threads in x-dimension) load a row of 4 floats
        int B_row = t * TILE_SIZE + threadIdx.y;
        if (threadIdx.x < (TILE_SIZE / 4)) {
            int col_start = blockIdx.x * TILE_SIZE + threadIdx.x * 4; // starting column for this vector load
            float4 b_vals;
            if (B_row < K && (col_start + 3) < N) {
                // Since B is row-major and N is divisible by 4, we can reinterpret cast B
                int vec_index = B_row * (N / 4) + (blockIdx.x * (TILE_SIZE / 4) + threadIdx.x);
                b_vals = ((const float4*)B)[vec_index];
            } else {
                b_vals.x = b_vals.y = b_vals.z = b_vals.w = 0.0f;
            }
            int store_col = threadIdx.x * 4;
            tileB[threadIdx.y][store_col]     = b_vals.x;
            tileB[threadIdx.y][store_col + 1] = b_vals.y;
            tileB[threadIdx.y][store_col + 2] = b_vals.z;
            tileB[threadIdx.y][store_col + 3] = b_vals.w;
        }

        // Synchronize to make sure the tile is completely loaded
        __syncthreads();

        // Compute the dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function wrapping the kernel launch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Tensor B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Ensure that N is divisible by 4 for safe vectorized loads
    TORCH_CHECK((N % 4) == 0, "N must be divisible by 4 for aligned vectorized load kernel");

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with aligned vectorized load (CUDA)");
}
