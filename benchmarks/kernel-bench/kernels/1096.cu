#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32 // Higher tile dimension for better memory utilization
#ifndef BLOCK_DIM
#define BLOCK_DIM 16
#endif

// Combined kernel that dynamically selects shared memory and warp size based on input dimensions
// Uses unified approach ensuring memory coalescing and a balance between tile size
// and block size for optimal performance

template <typename scalar_t>
__global__ void optimized_hybrid_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    int block_dim = (M > 1024) ? TILE_DIM : BLOCK_DIM;
    int global_row = blockIdx.y * block_dim + threadIdx.y;
    int col = blockIdx.x * block_dim + threadIdx.x;

    int batch = global_row / M;
    int m = global_row % M;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;
    int numTiles = (K + block_dim - 1) / block_dim;

    for (int t = 0; t < numTiles; ++t) {
        int A_col = t * block_dim + threadIdx.x;
        int B_row = t * block_dim + threadIdx.y;

        if (global_row < N * M && A_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = __ldg(&A[batch * M * K + m * K + A_col]);
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (B_row < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = __ldg(&B[B_row * L + col]);
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < block_dim; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (global_row < N * M && col < L) {
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

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "optimized_hybrid_kernel", ([&] {
        optimized_hybrid_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in optimized_hybrid_kernel: %s\n", cudaGetErrorString(err));
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
    m.def("forward", &module_fn_forward, "Optimized hybrid tensor-matrix multiplication (CUDA)");
}
