#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// This kernel uses grid-stride loops to allow each block to compute multiple output tiles.
// Each block loads sub-tiles from matrix A and B into shared memory and iterates over the K dimension in tiles.
// The outer loops (over row_tile and col_tile) use stride loops to cover the entire matrix dimensions, ensuring proper boundary handling.

template <typename scalar_t>
__global__ void matmul_stride_kernel(const scalar_t* __restrict__ A, 
                                       const scalar_t* __restrict__ B, 
                                       scalar_t* __restrict__ C,
                                       int M, int K, int N) {
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Loop over the output tiles using grid-stride loops in both row and column dimensions
    for (int row_tile = blockIdx.y * TILE_WIDTH; row_tile < M; row_tile += gridDim.y * TILE_WIDTH) {
        for (int col_tile = blockIdx.x * TILE_WIDTH; col_tile < N; col_tile += gridDim.x * TILE_WIDTH) {
            scalar_t value = 0;
            // Compute the global row and col for the current element computed by this thread
            int row = row_tile + ty;
            int col = col_tile + tx;

            // Number of tiles along the K dimension
            int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
            
            for (int t = 0; t < numTiles; ++t) {
                // Allocate shared memory for tiles of A and B
                __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
                __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];
                
                // Compute indices for the elements to load
                int A_row = row;
                int A_col = t * TILE_WIDTH + tx;
                int B_row = t * TILE_WIDTH + ty;
                int B_col = col;
                
                // Load tile from A into shared memory, with boundary check
                if (A_row < M && A_col < K)
                    sA[ty][tx] = A[A_row * K + A_col];
                else
                    sA[ty][tx] = static_cast<scalar_t>(0);
                
                // Load tile from B into shared memory, with boundary check
                if (B_row < K && B_col < N)
                    sB[ty][tx] = B[B_row * N + B_col];
                else
                    sB[ty][tx] = static_cast<scalar_t>(0);
                
                __syncthreads();
                
                // Compute partial product for this tile
                #pragma unroll
                for (int i = 0; i < TILE_WIDTH; ++i) {
                    value += sA[ty][i] * sB[i][tx];
                }
                
                __syncthreads();
            }
            
            // Write the result to C if within bounds
            if (row < M && col < N) {
                C[row * N + col] = value;
            }
        }
    }
}

// Host function exposed to Python via Pybind11

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");
    
    auto C = torch::empty({M, N}, A.options());
    
    // Launch a modest grid size; the kernel uses stride loops to cover the entire output matrix
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks( min((N + TILE_WIDTH - 1) / TILE_WIDTH, 32), 
                 min((M + TILE_WIDTH - 1) / TILE_WIDTH, 32) );
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_stride_kernel", ([&] {
        matmul_stride_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));
    
    cudaDeviceSynchronize();
    return C;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Stride loop tiled matrix multiplication");
}
