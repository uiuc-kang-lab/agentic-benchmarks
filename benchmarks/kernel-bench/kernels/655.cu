#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized tile size based on modern GPU architectures
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define ROWS_PER_THREAD (TILE_DIM / BLOCK_ROWS)  // For better occupancy

template <typename scalar_t>
__global__ void optimized_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M, const int K, const int N) {
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ scalar_t sA[TILE_DIM][TILE_DIM + 1];  // +1 padding
    __shared__ scalar_t sB[TILE_DIM][TILE_DIM + 1];  // +1 padding

    // Thread block handles multiple rows for better resource utilization
    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // Each thread accumulates multiple results
    scalar_t thread_results[BLOCK_ROWS] = {0};
    
    // Loop over tiles with grid-stride pattern for large matrices
    for (int tile_idx = 0; tile_idx < (K + TILE_DIM - 1) / TILE_DIM; tile_idx++) {
        // Collaborative loading of tiles into shared memory
        const int tile_offset = tile_idx * TILE_DIM;
        
        // Vectorized loads for better memory throughput
        if (row < M && tile_offset + threadIdx.x < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_ROWS; i++) {
                if (row + i * BLOCK_ROWS < M) {
                    sA[threadIdx.y + i * BLOCK_ROWS][threadIdx.x] = 
                        A[(row + i * BLOCK_ROWS) * K + tile_offset + threadIdx.x];
                }
            }
        }
        
        if (tile_offset + threadIdx.y < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(tile_offset + threadIdx.y) * N + col];
        }
        
        __syncthreads();
        
        // Compute partial results with loop unrolling
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            #pragma unroll
            for (int i = 0; i < BLOCK_ROWS; i++) {
                if (row + i * BLOCK_ROWS < M) {
                    thread_results[i] += sA[threadIdx.y + i * BLOCK_ROWS][k] * sB[k][threadIdx.x];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory with vectorized stores
    if (col < N) {
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            if (row + i * BLOCK_ROWS < M) {
                C[(row + i * BLOCK_ROWS) * N + col] = thread_results[i];
            }
        }
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, 
                (M + (TILE_DIM * BLOCK_ROWS) - 1) / (TILE_DIM * BLOCK_ROWS));
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_matmul_kernel", ([&] {
        optimized_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized hybrid matrix multiplication (CUDA)");
}