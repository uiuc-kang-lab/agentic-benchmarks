#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Tile dimension for shared memory tiling
#define TILE_DIM 32

// This kernel minimizes warp divergence by refactoring conditional logic. 
// Each block checks once if it is a full tile (i.e., all threads are within valid bounds) and then
// performs unconditional loads from global memory. For boundary blocks, conditional (ternary) operations
// are used. This uniform control flow reduces divergent branches within warps when processing full tiles.

template <typename scalar_t>
__global__ void branchless_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N,  // batch dimension
    const int M,  // matrix row dimension
    const int K,  // common dimension
    const int L   // matrix column dimension
) {
    // Flatten the (N x M) output into a 2D matrix of dimensions (N*M) x L
    int global_row = blockIdx.y * TILE_DIM + threadIdx.y; // row index in flattened A
    int global_col = blockIdx.x * TILE_DIM + threadIdx.x; // column index in output

    // Check if the entire block falls within valid output bounds
    bool full_tile = ((blockIdx.y * TILE_DIM + TILE_DIM) <= (N * M)) && ((blockIdx.x * TILE_DIM + TILE_DIM) <= L);

    __shared__ scalar_t sA[TILE_DIM][TILE_DIM];
    __shared__ scalar_t sB[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    
    // Loop over tiles along the K dimension
    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_DIM + threadIdx.x; // column index in A
        int B_row = t * TILE_DIM + threadIdx.y; // row index in B

        if (full_tile) {
            // In full tiles, no boundary checks are needed, allowing uniform control flow.
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[global_row * K + A_col]);
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[B_row * L + global_col]);
        } else {
            // For boundary tiles, use ternary operator to avoid divergent branches within warps
            sA[threadIdx.y][threadIdx.x] = (global_row < (N * M) && A_col < K) ? __ldg(&A[global_row * K + A_col]) : scalar_t(0);
            sB[threadIdx.y][threadIdx.x] = (B_row < K && global_col < L) ? __ldg(&B[B_row * L + global_col]) : scalar_t(0);
        }
        __syncthreads();

        // Compute partial dot product for the tile
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result back if within valid bounds
    if (full_tile) {
        output[global_row * L + global_col] = sum;
    } else {
        if (global_row < (N * M) && global_col < L)
            output[global_row * L + global_col] = sum;
    }
}

// CUDA forward function to launch the branchless_tiled_kernel
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    // The output tensor is viewed as a matrix of dimensions (N*M) x L
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "branchless_tiled_kernel", ([&] {
        branchless_tiled_kernel<scalar_t><<<grid, threads>>>(
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

// Input checking macros
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface that exposes the forward function to Pybind11
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
    m.def("forward", &module_fn_forward, "Branchless tiled tensor-matrix multiplication (CUDA)");
}
