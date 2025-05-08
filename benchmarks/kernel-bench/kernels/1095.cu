#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32

// This kernel performs 3D tensor-matrix multiplication while ensuring memory coalescing.
// It arranges global memory accesses so that threads in a warp read/write consecutive memory locations.
// Global loads use __ldg to take advantage of the read-only cache. Shared memory tiling is employed
// to reuse data and reduce global memory transactions.

template <typename scalar_t>
__global__ void coalesced_tiled_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    // Compute the global row index in the flattened (N*M) output.
    int global_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Map the flattened row to batch and row indices for A
    int batch = global_row / M;
    int m = global_row % M;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_DIM + threadIdx.x;  // Column index in A
        int B_row = t * TILE_DIM + threadIdx.y;   // Row index in B

        // Load element from A ensuring coalesced access along rows
        if (batch < N && m < M && A_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = __ldg(&A[batch * M * K + m * K + A_col]);
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        // Load element from B ensuring coalesced access along rows
        if (B_row < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = __ldg(&B[B_row * L + col]);
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute dot product for the current tile
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the output if within bounds; writes are coalesced along the row dimension
    if (batch < N && m < M && col < L) {
        output[batch * M * L + m * L + col] = sum;
    }
}


void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    // The output is treated as a 2D array with (N*M) rows and L columns
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "coalesced_tiled_kernel", ([&] {
        coalesced_tiled_kernel<scalar_t><<<grid, threads>>>(
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

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
    m.def("forward", &module_fn_forward, "Coalesced tiled tensor-matrix multiplication (CUDA)");
}
