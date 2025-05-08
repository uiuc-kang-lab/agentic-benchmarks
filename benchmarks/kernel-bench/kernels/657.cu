#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define ELEMENTS_PER_THREAD 4  // Each thread processes multiple elements

template <typename scalar_t>
__global__ void strided_matmul_kernel(const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C,
                                     const int M, const int K, const int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];
    
    // Calculate base indices
    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int block_row = blockIdx.y * TILE_WIDTH;
    const int block_col = blockIdx.x * (TILE_WIDTH * ELEMENTS_PER_THREAD);

    // Grid stride for better work distribution
    for (int row_offset = 0; row_offset < M; row_offset += gridDim.y * TILE_WIDTH) {
        const int global_row = block_row + row_offset + thread_row;
        
        // Process multiple columns per thread using stride
        for (int col_stride = 0; col_stride < ELEMENTS_PER_THREAD; col_stride++) {
            scalar_t accumulator = 0;
            
            // Tile multiplication loop
            for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
                // Load tile of A into shared memory
                if (global_row < M && (t * TILE_WIDTH + thread_col) < K) {
                    sA[thread_row][thread_col] = A[global_row * K + t * TILE_WIDTH + thread_col];
                } else {
                    sA[thread_row][thread_col] = 0;
                }

                // Load tile of B into shared memory
                const int global_col = block_col + col_stride * TILE_WIDTH + thread_col;
                if ((t * TILE_WIDTH + thread_row) < K && global_col < N) {
                    sB[thread_row][thread_col] = B[(t * TILE_WIDTH + thread_row) * N + global_col];
                } else {
                    sB[thread_row][thread_col] = 0;
                }

                __syncthreads();

                // Compute partial dot product
                #pragma unroll
                for (int k = 0; k < TILE_WIDTH; k++) {
                    accumulator += sA[thread_row][k] * sB[k][thread_col];
                }

                if (t < (K + TILE_WIDTH - 1) / TILE_WIDTH - 1) {
                    __syncthreads();
                }
            }

            // Store result
            const int global_col = block_col + col_stride * TILE_WIDTH + thread_col;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accumulator;
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

    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    // Adjust grid dimensions to account for multiple elements per thread
    const int grid_cols = (N + (TILE_WIDTH * ELEMENTS_PER_THREAD) - 1) / (TILE_WIDTH * ELEMENTS_PER_THREAD);
    const int grid_rows = (M + TILE_WIDTH - 1) / TILE_WIDTH;

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(grid_cols, grid_rows);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "strided_matmul_kernel", ([&] {
        strided_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Strided matrix multiplication forward (CUDA)");
}