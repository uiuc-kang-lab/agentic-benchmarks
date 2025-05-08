#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tile dimension for shared memory tiling
#define TILE_DIM 32

// Optimized kernel: combines coalesced memory accesses, shared memory tiling, and __ldg read-only cache
template <typename scalar_t>
__global__ void optimized_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    // We flatten the first two dimensions (batch and row) of A into a single dimension of size (N*M).
    // Thus, we view A as a 2D matrix of dimensions (N*M) x K and output as (N*M) x L.
    int row = blockIdx.y * TILE_DIM + threadIdx.y; // Flattened row index (n * M + m)
    int col = blockIdx.x * TILE_DIM + threadIdx.x; // Column index for output

    scalar_t sum = 0;

    // Shared memory tiles for A and B
    __shared__ scalar_t sA[TILE_DIM][TILE_DIM];
    __shared__ scalar_t sB[TILE_DIM][TILE_DIM];

    // Loop over tiles in the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_DIM + threadIdx.x; // Column index in A
        int B_row = t * TILE_DIM + threadIdx.y;   // Row index in B

        // Load a tile of A into shared memory
        if (row < (N * M) && A_col < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + A_col]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // Load a tile of B into shared memory; B is of shape (K, L)
        if (B_row < K && col < L)
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[B_row * L + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Compute partial dot-product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed output if within the valid bounds
    if (row < (N * M) && col < L)
        output[row * L + col] = sum;
}

// CUDA forward function that dispatches the optimized kernel
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    // Define block and grid dimensions based on tile size
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "optimized_tiled_kernel", ([&] {
        optimized_tiled_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// Macros for input checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface function
torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int N = A.size(0);
    const int M = A.size(1);
    const int L = B.size(1);

    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Efficient tiled tensor-matrix multiplication (CUDA)");
}
