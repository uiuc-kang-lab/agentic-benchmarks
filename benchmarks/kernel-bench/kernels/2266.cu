#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile size for shared memory tiling
#define TILE_SIZE 16

// CUDA kernel for computing C = A.T * B using optimized shared memory tiling.
// A: shape (K, M), B: shape (K, N), C: shape (M, N).
// Note: A.T(i,k) = A(k,i), so we load A in a transposed manner from global memory.
__global__ void matMulOptimizedSharedKernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int K, int M, int N) {
    // Compute the row (i) and column (j) index in the output matrix C
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;  // corresponds to i in C (and A's column index)
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;  // corresponds to j in C (and B's column index)

    float sum = 0.0f;

    // Allocate shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Use double-buffering to overlap tile loading with computation
    const int bufferCount = 2;
    __shared__ float tileA_buf[bufferCount][TILE_SIZE][TILE_SIZE];
    __shared__ float tileB_buf[bufferCount][TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int curr_buf = 0;

    // Prefetch the first tile into buffer 0
    {
        int t = 0;
        int aIndex = t * TILE_SIZE + threadIdx.x;
        int bIndex = t * TILE_SIZE + threadIdx.y;
        if (row < M && aIndex < K) {
            tileA_buf[curr_buf][threadIdx.y][threadIdx.x] = A[aIndex * M + row];
        } else {
            tileA_buf[curr_buf][threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (bIndex < K && col < N) {
            tileB_buf[curr_buf][threadIdx.y][threadIdx.x] = B[bIndex * N + col];
        } else {
            tileB_buf[curr_buf][threadIdx.y][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    // Pipeline: for each tile, prefetch next tile while computing current tile
    for (int t = 0; t < numTiles - 1; t++) {
        int next_buf = 1 - curr_buf;
        // Prefetch next tile into next_buf
        {
            int aIndex = (t + 1) * TILE_SIZE + threadIdx.x;
            int bIndex = (t + 1) * TILE_SIZE + threadIdx.y;
            if (row < M && aIndex < K) {
                tileA_buf[next_buf][threadIdx.y][threadIdx.x] = A[aIndex * M + row];
            } else {
                tileA_buf[next_buf][threadIdx.y][threadIdx.x] = 0.0f;
            }
            if (bIndex < K && col < N) {
                tileB_buf[next_buf][threadIdx.y][threadIdx.x] = B[bIndex * N + col];
            } else {
                tileB_buf[next_buf][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();  // Ensure next tile is loaded

        // Compute on the current tile
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
            sum += tileA_buf[curr_buf][threadIdx.y][k_inner] * tileB_buf[curr_buf][k_inner][threadIdx.x];
        }

        __syncthreads();  // Ensure computation is done before buffer swap
        curr_buf = next_buf;  // Swap buffers for next iteration
    }

    // Process the last tile (prefetched already)
    #pragma unroll
    for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
        sum += tileA_buf[curr_buf][threadIdx.y][k_inner] * tileB_buf[curr_buf][k_inner][threadIdx.x];
    }

    // Write the computed value to C if within valid indices
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Dimensions: A is (K, M) and B is (K, N).
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes using tiling.
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Get raw pointers to tensor data.
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the optimized shared memory tiled CUDA kernel.
    matMulOptimizedSharedKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using optimized shared memory tiling");
}
