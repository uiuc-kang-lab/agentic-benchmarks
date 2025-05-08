#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32

// This kernel performs 3D tensor-matrix multiplication for A (N x M x K) and B (K x L).
// The output is a tensor of shape (N x M x L) computed by flattening the first two dimensions of A into (N*M) x K.
// To minimize loop overhead, critical loops are unrolled with #pragma unroll. The kernel uses shared memory
// tiling with coalesced global memory loads (with __ldg) and branchless boundary handling when possible.

template <typename scalar_t>
__global__ void unrolled_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    // Flatten the (N, M) output indices into one: each row corresponds to a combined index (n*M + m)
    int global_row = blockIdx.y * TILE_DIM + threadIdx.y;  // Index into flattened A: [N*M x K]
    int global_col = blockIdx.x * TILE_DIM + threadIdx.x;  // Column index in output, corresponds to B's column

    // Check if entire block lies inside valid boundaries
    bool full_tile = ((blockIdx.y * TILE_DIM + TILE_DIM) <= (N * M)) && ((blockIdx.x * TILE_DIM + TILE_DIM) <= L);

    __shared__ scalar_t sA[TILE_DIM][TILE_DIM];
    __shared__ scalar_t sB[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Loop over tiles in the K dimension; unroll for reduced loop overhead when possible
    #pragma unroll
    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_DIM + threadIdx.x;  // Column index inside A
        int B_row = t * TILE_DIM + threadIdx.y;    // Row index inside B

        // Load tile from A and B into shared memory with minimal divergence
        if (full_tile) {
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[global_row * K + A_col]);
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[B_row * L + global_col]);
        } else {
            sA[threadIdx.y][threadIdx.x] = (global_row < (N * M) && A_col < K) ? __ldg(&A[global_row * K + A_col]) : scalar_t(0);
            sB[threadIdx.y][threadIdx.x] = (B_row < K && global_col < L) ? __ldg(&B[B_row * L + global_col]) : scalar_t(0);
        }
        __syncthreads();

        // Compute partial dot-product for the current tile; unroll inner loop
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the output if inside valid boundaries
    if (global_row < (N * M) && global_col < L) {
        output[global_row * L + global_col] = sum;
    }
}

// CUDA forward function to launch the unrolled_tiled_kernel
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    // Define block and grid dimensions. The output is treated as a matrix of dimensions (N*M x L)
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "unrolled_tiled_kernel", ([&] {
        unrolled_tiled_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &module_fn_forward, "Unrolled tiled tensor-matrix multiplication (CUDA)");
}
