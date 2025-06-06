#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel performs lower-triangular matrix multiplication C = A * B, where A and B are lower triangular.
// It uses tiled shared memory loading and unrolls the inner summation loop to reduce loop overhead.

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Only compute for the lower triangular portion
    if (row < N && col < N && row >= col) {
        // Loop over tiles. Each tile corresponds to a segment of the summation index.
        for (int k_tile = blockIdx.x; k_tile <= blockIdx.y; ++k_tile) {
            int tiled_row = k_tile * TILE_SIZE + threadIdx.y;
            int tiled_col = k_tile * TILE_SIZE + threadIdx.x;

            // Load A tile: Only load if within bounds and if within the lower triangular limits
            A_tile[threadIdx.y][threadIdx.x] = ((row < N) && (tiled_col < N) && (tiled_col <= row)) ?
                                               A[row * N + tiled_col] : 0.0f;
            
            // Load B tile: Only load if within bounds and if column index is valid
            if ((tiled_row < N) && (col < N) && (col <= tiled_row))
                B_tile[threadIdx.y][threadIdx.x] = B[tiled_row * N + col];
            else
                B_tile[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();

            // Unroll the inner loop to reduce overhead
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = k_tile * TILE_SIZE + k;
                if (global_k >= col && global_k <= row) {
                    sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
                }
            }

            __syncthreads();
        }
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with unrolled inner loop (CUDA)");
}
