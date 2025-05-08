#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define VECTOR_SIZE 4  // Using float4 for vectorized loads

template <typename scalar_t>
__global__ void vectorized_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * (TILE_DIM * VECTOR_SIZE) + threadIdx.x * VECTOR_SIZE;
    const int batch_idx = row / M;
    const int m_idx = row % M;

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM + VECTOR_SIZE];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM + VECTOR_SIZE];

    scalar_t sum[VECTOR_SIZE] = {0};

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        if (batch_idx < N && m_idx < M) {
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                int k_idx = t * TILE_DIM + threadIdx.x * VECTOR_SIZE + v;
                if (k_idx < K) {
                    tile_A[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 
                        A[batch_idx * M * K + m_idx * K + k_idx];
                } else {
                    tile_A[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0;
                }
            }
        }

        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            int k_idx = t * TILE_DIM + threadIdx.y;
            int l_idx = col + v;
            if (k_idx < K && l_idx < L) {
                tile_B[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 
                    B[k_idx * L + l_idx];
            } else {
                tile_B[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            scalar_t a_val = tile_A[threadIdx.y][k];
            #pragma unroll
            for (int v = 0; v < VECTOR_SIZE; v++) {
                sum[v] += a_val * tile_B[k][threadIdx.x * VECTOR_SIZE + v];
            }
        }

        __syncthreads();
    }

    if (batch_idx < N && m_idx < M) {
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            int l_idx = col + v;
            if (l_idx < L) {
                output[batch_idx * M * L + m_idx * L + l_idx] = sum[v];
            }
        }
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

    dim3 threads(TILE_DIM / VECTOR_SIZE, TILE_DIM);
    dim3 grid((L + TILE_DIM * VECTOR_SIZE - 1) / (TILE_DIM * VECTOR_SIZE), 
              (N * M + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "vectorized_kernel", ([&] {
        vectorized_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &module_fn_forward, "Vectorized tensor-matrix multiplication (CUDA)");
}