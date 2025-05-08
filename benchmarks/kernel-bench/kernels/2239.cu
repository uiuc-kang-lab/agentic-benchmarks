/*
This CUDA kernel computes C = A.T * B where A is of shape (K, M) and B is of shape (K, N).
It combines shared memory tiling (to improve memory reuse) and the split-K technique (to expose parallelism along the K dimension).
A templated parameter SPLIT_K allows the kernel to avoid atomic operations when no K-splitting is needed (i.e. SPLIT_K == 1) and use atomics only when splitting, reducing overhead.

Compile with a C++ compiler that supports CUDA and PyBind11.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimension for shared memory tiling
#define TILE_DIM 16

// Templated kernel: SPLIT_K is a compile time constant to tailor the execution
// For SPLIT_K == 1, the kernel writes its result directly; for SPLIT_K > 1, each block computes a partial result and atomicAdd is used to accumulate into C.

template <int SPLIT_K>
__global__ void tiledSplitKKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int K, int M, int N) {
    // Determine K-range for this block based on the split factor
    int block_k_size = (K + SPLIT_K - 1) / SPLIT_K;
    int k_start = blockIdx.z * block_k_size;
    int k_end = (k_start + block_k_size < K) ? (k_start + block_k_size) : K;

    // Compute output row and column indices (C is of shape M x N, computed as A.T * B)
    int row = blockIdx.x * TILE_DIM + threadIdx.y;  // corresponds to a row in C and a column in A
    int col = blockIdx.y * TILE_DIM + threadIdx.x;  // corresponds to a column in C and B

    float cValue = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int local_k = k_end - k_start;
    int numTiles = (local_k + TILE_DIM - 1) / TILE_DIM;

    // Loop over tiles along the local K dimension
    for (int t = 0; t < numTiles; t++) {
        int k_idx = t * TILE_DIM + threadIdx.x;
        int global_k = k_start + k_idx;
        if (row < M && global_k < k_end) {
            // A is stored as (K, M): element A[global_k, row] is at A[global_k * M + row]
            As[threadIdx.y][threadIdx.x] = A[global_k * M + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int k_idx_b = t * TILE_DIM + threadIdx.y;
        global_k = k_start + k_idx_b;
        if (col < N && global_k < k_end) {
            // B is stored as (K, N): element B[global_k, col] is at B[global_k * N + col]
            Bs[threadIdx.y][threadIdx.x] = B[global_k * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the tiles
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            cValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to C. If SPLIT_K > 1, accumulate using atomicAdd.
    if (row < M && col < N) {
        if (SPLIT_K == 1) {
            C[row * N + col] = cValue;
        } else {
            atomicAdd(&C[row * N + col], cValue);
        }
    }
}

// The forward function exposed via PyBind11. An extra parameter 'split_k' controls the split factor along the K dimension.
// For split_k == 1, the kernel operates without atomics. For larger values, the K dimension is partitioned among multiple blocks.

torch::Tensor forward(torch::Tensor A, torch::Tensor B, int split_k) {
    // Ensure inputs are CUDA tensors and of type float32
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A: (K, M), B: (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C, initialized to zero. When using split-K > 1, we accumulate partial results into C.
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    int gridX = (M + TILE_DIM - 1) / TILE_DIM;
    int gridY = (N + TILE_DIM - 1) / TILE_DIM;
    int gridZ = split_k;  // gridDim.z controls the split along the K dimension
    dim3 gridDim(gridX, gridY, gridZ);

    // Launch the kernel with a compile-time SPLIT_K based on the provided split_k value.
    // Only a few discrete split_k values are supported here; fallback to split_k = 2 if an unsupported value is provided.
    if (split_k == 1) {
        tiledSplitKKernel<1><<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    } else if (split_k == 2) {
        tiledSplitKKernel<2><<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    } else if (split_k == 4) {
        tiledSplitKKernel<4><<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    } else if (split_k == 8) {
        tiledSplitKKernel<8><<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    } else {
        // Fallback to a default split factor (e.g., 2) if an unsupported split_k is provided
        tiledSplitKKernel<2><<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient computation of C = A.T * B using tiling and split-K (CUDA)");
}
