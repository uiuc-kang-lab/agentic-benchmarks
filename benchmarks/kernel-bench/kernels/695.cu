#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

template <typename scalar_t>
__global__ void uniform_control_flow_matmul(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N) {
    
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    const bool row_valid = row < M;
    const bool col_valid = col < N;
    
    scalar_t value = 0;

    const int num_full_tiles = K / TILE_WIDTH;
    const int partial_tile_size = K % TILE_WIDTH;

    // Process full tiles without bounds checking
    for (int t = 0; t < num_full_tiles; ++t) {
        const int a_col = t * TILE_WIDTH + threadIdx.x;
        const int b_row = t * TILE_WIDTH + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = row_valid ? A[row * K + a_col] : 0;
        sB[threadIdx.y][threadIdx.x] = col_valid ? B[b_row * N + col] : 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Process partial tile with masked operations
    if (partial_tile_size > 0) {
        const int a_col = num_full_tiles * TILE_WIDTH + threadIdx.x;
        const int b_row = num_full_tiles * TILE_WIDTH + threadIdx.y;
        
        const bool a_valid = row_valid && (a_col < K);
        const bool b_valid = col_valid && (b_row < K);

        sA[threadIdx.y][threadIdx.x] = a_valid ? A[row * K + a_col] : 0;
        sB[threadIdx.y][threadIdx.x] = b_valid ? B[b_row * N + col] : 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Single conditional write per thread
    if (row_valid && col_valid) {
        C[row * N + col] = value;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions mismatch");

    auto C = torch::zeros({M, N}, A.options());

    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "uniform_control_flow_matmul", ([&] {
        uniform_control_flow_matmul<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Uniform control flow matrix multiplication");
}
