#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_L 16
#define TILE_K 16

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {
    
    __shared__ scalar_t As[TILE_M][TILE_K];
    __shared__ scalar_t Bs[TILE_K][TILE_L];

    int n = blockIdx.z;
    int m_tile = blockIdx.y * TILE_M;
    int l_tile = blockIdx.x * TILE_L;
    
    int thread_m = threadIdx.y;
    int thread_l = threadIdx.x;

    scalar_t sum = 0;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile
        if (m_tile + thread_m < M && k_tile + thread_l < K) {
            As[thread_m][thread_l] = A[n * M * K + (m_tile + thread_m) * K + (k_tile + thread_l)];
        } else {
            As[thread_m][thread_l] = 0;
        }

        // Load B tile
        if (k_tile + thread_m < K && l_tile + thread_l < L) {
            Bs[thread_m][thread_l] = B[n * K * L + (k_tile + thread_m) * L + (l_tile + thread_l)];
        } else {
            Bs[thread_m][thread_l] = 0;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_K; k++) {
            sum += As[thread_m][k] * Bs[k][thread_l];
        }
        __syncthreads();
    }

    if (n < N && m_tile + thread_m < M && l_tile + thread_l < L) {
        output[n * M * L + (m_tile + thread_m) * L + (l_tile + thread_l)] = sum;
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    dim3 blocks((L + TILE_L - 1) / TILE_L,
                (M + TILE_M - 1) / TILE_M,
                N);
    dim3 threads(TILE_L, TILE_M);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
}

TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cuda_forward);
}
