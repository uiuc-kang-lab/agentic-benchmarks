/*
 * Combined CUDA kernel for 3D tensor-matrix multiplication.
 *
 * This kernel multiplies a 3D tensor A of dimensions [N, M, K] by a matrix B of dimensions [K, L]
 * to produce an output tensor of dimensions [N, M, L].
 * It combines optimizations from two approaches:
 *   - Read-only cache usage via __ldg for efficient global memory loads (from Kernel 1).
 *   - Configurable block/tile dimension and shared memory tiling (from Kernel 2).
 *
 * The design ensures coalesced global loads/stores and reuse via shared memory tiling, reducing
 * global memory transactions, while the BLOCK_DIM parameter offers tuning flexibility.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Use BLOCK_DIM macro for tile dimension. Default to 32 if not defined.
#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

// Combined kernel using shared memory tiling with read-only __ldg loads.
template <typename scalar_t>
__global__ void coalesced_experiment_kernel(
    const scalar_t * __restrict__ A,
    const scalar_t * __restrict__ B,
    scalar_t * __restrict__ output,
    const int N, const int M, const int K, const int L) {

    const int block_dim = BLOCK_DIM;  // Block (tile) dimension
    // Calculate the flattened global row index. The output is treated as (N*M) x L.
    int global_row = blockIdx.y * block_dim + threadIdx.y;  // corresponds to (n*M + m)
    int col = blockIdx.x * block_dim + threadIdx.x;         // output column index in L dimension

    // Map the flattened row index to batch (n) and intra-batch row (m) indices.
    int batch = global_row / M;
    int m = global_row % M;

    // Allocate shared memory for tiles from A and B
    __shared__ scalar_t tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ scalar_t tile_B[BLOCK_DIM][BLOCK_DIM];

    scalar_t sum = 0;

    // Loop over tiles along the K dimension
    // Each iteration loads a tile of A and B into shared memory and computes partial dot products.
    for (int t = 0; t < K; t += block_dim) {
        int A_col = t + threadIdx.x;  // Column index for A
        int B_row = t + threadIdx.y;  // Row index for B

        // Load tile element from A using __ldg for read-only caching
        if (batch < N && m < M && A_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = __ldg(&A[batch * M * K + m * K + A_col]);
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile element from B using __ldg for read-only caching
        if (B_row < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = __ldg(&B[B_row * L + col]);
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < block_dim; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed output value if within valid bounds
    if (batch < N && m < M && col < L) {
        output[batch * M * L + m * L + col] = sum;
    }
}

// CUDA forward launch function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    int block_dim = BLOCK_DIM;
    // Grid configuration: output tensor is viewed as a 2D array of shape ((N*M) x L)
    dim3 threads(block_dim, block_dim);
    dim3 grid((L + block_dim - 1) / block_dim, ((N * M) + block_dim - 1) / block_dim);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "coalesced_experiment_kernel", ([&] {
        coalesced_experiment_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in coalesced_experiment_kernel: %s\n", cudaGetErrorString(err));
    }
}

// Macros for input tensor checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface to launch the CUDA kernel
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
    m.def("forward", &module_fn_forward, "Coalesced experiment tensor-matrix multiplication (CUDA)");
}
