#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// Define tile dimension
#define TILE_DIM 16

// CUDA kernel: each block computes a TILE_DIM x TILE_DIM tile of the output matrix for one slice of the 3D tensor.
// The 3D tensor A has shape (N, M, K) and B has shape (K, L). The output is (N, M, L).
// We map blockIdx.z to the batch dimension (n), blockIdx.y to the M dimension, and blockIdx.x to the L dimension.

template <typename scalar_t>
__global__ void matrix_mul_3d_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    // Determine indices: n is the batch index, m and l index the output 2D matrix for that slice.
    int n = blockIdx.z;
    int m = blockIdx.y * TILE_DIM + threadIdx.y;
    int l = blockIdx.x * TILE_DIM + threadIdx.x;

    // Only consider valid indices
    if (n < N && m < M && l < L) {
        scalar_t sum = 0;
        
        // Loop over tiles along the K dimension
        int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
        
        // Declare shared memory for tiles of A and B
        __shared__ scalar_t As[TILE_DIM][TILE_DIM];
        __shared__ scalar_t Bs[TILE_DIM][TILE_DIM];
        
        for (int t = 0; t < numTiles; t++) {
            int a_col = t * TILE_DIM + threadIdx.x; // Column index in A for current tile
            int b_row = t * TILE_DIM + threadIdx.y; // Row index in B for current tile
            
            // Load tile element from A (from the slice corresponding to n)
            if (m < M && a_col < K) {
                As[threadIdx.y][threadIdx.x] = A[n * M * K + m * K + a_col];
            } else {
                As[threadIdx.y][threadIdx.x] = 0;
            }
            
            // Load tile element from B
            if (b_row < K && l < L) {
                Bs[threadIdx.y][threadIdx.x] = B[b_row * L + l];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0;
            }
            
            __syncthreads();
            
            // Compute partial dot product for this tile
            #pragma unroll
            for (int i = 0; i < TILE_DIM; i++) {
                sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        // Write the result to the output tensor
        output[n * M * L + m * L + l] = sum;
    }
}

// CUDA forward function
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    // Set up 3D grid: grid.x covers the L dimension, grid.y covers the M dimension, grid.z covers the N (batch) dimension
    dim3 block(TILE_DIM, TILE_DIM, 1);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM, N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "matrix_mul_3d_kernel", ([&] {
        matrix_mul_3d_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    // Check for errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);

    // Allocate output tensor with shape (N, M, L)
    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "3D Tensor-Matrix Multiplication with 3D grid (CUDA)");
}
