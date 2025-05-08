#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a larger tile size to improve instruction level parallelism
#define TILE_WIDTH 32

// CUDA kernel for matrix multiplication using shared memory tiling, register blocking, and loop unrolling.
// Each thread computes one element of the output matrix, so no atomic operations are needed.

template <typename scalar_t>
__global__ void register_tile_matmul_kernel(const scalar_t* __restrict__ A, 
                                              const scalar_t* __restrict__ B,
                                              scalar_t* __restrict__ C,
                                              int M, int K, int N) {
    // Allocate shared memory for tiles of A and B
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    // Compute the row and column index of the C element to work on
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t value = 0;  // Accumulator in register

    // Loop over the tiles of A and B required to compute C(row, col)
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        // Load one element of the tile from A into shared memory
        int tiledACol = t * TILE_WIDTH + threadIdx.x;
        if (row < M && tiledACol < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + tiledACol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0;
        }

        // Load one element of the tile from B into shared memory
        int tiledBRow = t * TILE_WIDTH + threadIdx.y;
        if (tiledBRow < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[tiledBRow * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0;
        }

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Multiply the two tiles together, with loop unrolling to reduce loop overhead
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            /* Use fast math intrinsics for float precision if available */
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same<scalar_t, float>::value) {
                value = __fmaf_rn(sA[threadIdx.y][i], sB[i][threadIdx.x], value);
            } else {
                value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
            }
            #else
                value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
            #endif
        }

        // Synchronize to make sure that computation using the tile is done before loading a new tile
        __syncthreads();
    }

    // Write the computed value to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    // Ensure input tensors are CUDA tensors
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    // Get matrix dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    // Check dimensions compatibility
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    // Allocate output tensor
    auto C = torch::empty({M, N}, A.options());

    // Define block and grid dimensions
    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "register_tile_matmul_kernel", ([&] {
        register_tile_matmul_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    return C;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA)");
}
