#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void coalesced_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    __shared__ scalar_t s_A[TILE_DIM][TILE_DIM + 1];
    __shared__ scalar_t s_B[TILE_DIM][TILE_DIM + 1];

    const int batch_id = blockIdx.z;
    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int batch_offset = batch_id * M * K;
    const scalar_t* A_batch = A + batch_offset;
    scalar_t sum = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        const int k_offset = t * TILE_DIM;
        
        if (row < M) {
            #pragma unroll
            for (int i = 0; i < TILE_DIM; i += WARP_SIZE) {
                int k_idx = k_offset + tx + i;
                if (k_idx < K) {
                    s_A[ty][tx + i] = A_batch[row * K + k_idx];
                } else {
                    s_A[ty][tx + i] = 0;
                }
            }
        }

        if (k_offset + ty < K && col < L) {
            #pragma unroll
            for (int i = 0; i < TILE_DIM; i += WARP_SIZE) {
                int k_idx = k_offset + ty + i;
                if (k_idx < K) {
                    s_B[ty + i][tx] = B[k_idx * L + col];
                } else {
                    s_B[ty + i][tx] = 0;
                }
            }
        }

        __syncthreads();

        if (row < M && col < L) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k += 4) {
                sum += s_A[ty][k] * s_B[k][tx];
                sum += s_A[ty][k + 1] * s_B[k + 1][tx];
                sum += s_A[ty][k + 2] * s_B[k + 2][tx];
                sum += s_A[ty][k + 3] * s_B[k + 3][tx];
            }
        }

        __syncthreads();
    }

    if (row < M && col < L) {
        output[batch_id * M * L + row * L + col] = sum;
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
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, 
              (M + TILE_DIM - 1) / TILE_DIM,
              N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "coalesced_matmul_kernel", ([&] {
        coalesced_matmul_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &module_fn_forward, "Coalesced tensor-matrix multiplication (CUDA)");
}