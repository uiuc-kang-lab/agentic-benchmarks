#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 16
#define UNROLL_FACTOR 4

template <typename scalar_t>
__global__ void tiled_tensor_matrix_multiplication_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int num_rows,
    const int K,
    const int L) {

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    scalar_t sum = 0;

    #pragma unroll
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        if (row < num_rows && (t * TILE_DIM + threadIdx.x) < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }

        if ((t * TILE_DIM + threadIdx.y) < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * L + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Manual unrolling of the multiplication loop
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i += UNROLL_FACTOR) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
            sum += tile_A[threadIdx.y][i + 1] * tile_B[i + 1][threadIdx.x];
            sum += tile_A[threadIdx.y][i + 2] * tile_B[i + 2][threadIdx.x];
            sum += tile_A[threadIdx.y][i + 3] * tile_B[i + 3][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < num_rows && col < L) {
        output[row * L + col] = sum;
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
    const int num_rows = N * M;

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, (num_rows + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        tiled_tensor_matrix_multiplication_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_rows, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

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

    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}