#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel that uses grid-stride loops over output tiles
// Each block computes one or more TILE_WIDTH x TILE_WIDTH tiles of the output matrix C
template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A,
                                     const scalar_t* __restrict__ B,
                                     scalar_t* __restrict__ C,
                                     int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    // Use grid-stride loops to cover all output tiles
    // Each block starts at a tile determined by its blockIdx and then strides by gridDim
    for (int tile_row = blockIdx.y * TILE_WIDTH; tile_row < M; tile_row += gridDim.y * TILE_WIDTH) {
        for (int tile_col = blockIdx.x * TILE_WIDTH; tile_col < N; tile_col += gridDim.x * TILE_WIDTH) {
            // Each thread computes one element within the current tile
            int row = tile_row + threadIdx.y;
            int col = tile_col + threadIdx.x;
            scalar_t value = 0;

            // Loop over tiles in the K dimension
            int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
            for (int t = 0; t < num_tiles; t++) {
                int A_col = t * TILE_WIDTH + threadIdx.x;
                int B_row = t * TILE_WIDTH + threadIdx.y;

                // Load A tile element if within bounds
                if (row < M && A_col < K) 
                    sA[threadIdx.y][threadIdx.x] = (row < M && A_col < K) ? A[row * K + A_col] : 0;
                else 
                    sA[threadIdx.y][threadIdx.x] = 0;

                // Load B tile element if within bounds
                if (col < N && B_row < K)
                    sB[threadIdx.y][threadIdx.x] = B[B_row * N + col];
                else
                    sB[threadIdx.y][threadIdx.x] = 0;

                __syncthreads();

                // Multiply the two tiles together
                for (int i = 0; i < TILE_WIDTH; i++) {
                    value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
                }
                __syncthreads();
            }

            // Write the computed value to output matrix if within bounds
            if (row < M && col < N) {
                C[row * N + col] = value;
            }
        }
    }
}

// Forward function
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    // Define block dimensions
    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    // Set grid to cover the output tile; grid-stride loops will cover extra tiles if needed
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();

    return C;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA) with grid-stride loops");
}
