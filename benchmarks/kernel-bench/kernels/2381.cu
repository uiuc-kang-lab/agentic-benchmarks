#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory tiling
#define TILE_SIZE 16

// Optimized kernel with optional split-K: when split_k > 1, multiple blocks accumulate partial results with atomicAdd
__global__ void optimized_matmul_transposed_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* C,
                                                     int M, int N, int K,
                                                     int k_tile) {
    // Shared memory for A and B tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tz = blockIdx.z;  // used for split-K
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and col indices for C
    int m = by * TILE_SIZE + ty;
    int n = bx * TILE_SIZE + tx;

    // Determine the slice of the K dimension this block will work on
    int k_start = tz * k_tile;
    int k_end = (k_start + k_tile < K) ? (k_start + k_tile) : K;
    
    float c_val = 0.0f;

    // Compute number of tiles for this block's K-slice
    int numTiles = (k_tile + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int k_offset = k_start + t * TILE_SIZE;
        
        // Load tile from A: each thread loads one element if within bounds
        if (m < M && (k_offset + tx) < k_end) {
            As[ty][tx] = A[m * K + (k_offset + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B, taking into account the transposition
        if ((k_offset + ty) < k_end && n < N) {
            Bs[ty][tx] = B[(k_offset + ty) * N + n];  // Corrected indexing for transposed B
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k) {
            c_val += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write out the computed partial sum to global memory
    // When only one K-slice is used, there is no race condition, so direct store is safe.
    if (m < M && n < N) {
        if (gridDim.z == 1) {
            C[m * N + n] = c_val;
        } else {
            atomicAdd(&C[m * N + n], c_val);
        }
    }
}

// The forward function optionally accepts a split_k parameter to partition the K dimension
// When split_k == 1, atomic operations are avoided.
// When split_k > 1, the K dimension is split among gridDim.z blocks and atomicAdd is used to correctly accumulate results.

torch::Tensor forward(torch::Tensor A, torch::Tensor B, int split_k = 1) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    // For split-K accumulation, initialize C with zeros
    torch::Tensor C = (split_k > 1) ? torch::zeros({M, N}, A.options()) : torch::empty({M, N}, A.options());
    
    // Determine how many K elements each split gets
    int k_tile = (K + split_k - 1) / split_k;
    
    // Setup grid and block dimensions
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              split_k);
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    // Launch the kernel
    optimized_matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, k_tile);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix multiplication with transposed B (CUDA)");
}
