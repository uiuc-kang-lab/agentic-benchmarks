#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_M 32
#define TILE_L 8
#define TILE_K 32

template <typename scalar_t>
__global__ void shared_mem_tiling_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {
    
    __shared__ scalar_t As[TILE_M][TILE_K];
    __shared__ scalar_t Bs[TILE_K][TILE_L];

    int n = blockIdx.z;
    int m_tile = blockIdx.x * TILE_M;
    int l_tile = blockIdx.y * TILE_L;
    
    int thread_m = threadIdx.x;
    int thread_l = threadIdx.y;
    int m = m_tile + thread_m;
    int l = l_tile + thread_l;
    
    scalar_t sum = 0;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile
        int k_a = k_tile + thread_l;
        if (m < M && k_a < K) As[thread_m][thread_l] = A[n*M*K + m*K + k_a];
        else As[thread_m][thread_l] = 0;

        // Load B tile
        int k_b = k_tile + thread_m;
        if (k_b < K && l < L) Bs[thread_m][thread_l] = B[k_b*L + l];
        else Bs[thread_m][thread_l] = 0;

        __syncthreads();

        // Compute partial sum
        for (int kk = 0; kk < TILE_K; ++kk) {
            sum += As[thread_m][kk] * Bs[kk][thread_l];
        }
        __syncthreads();
    }

    if (m < M && l < L) {
        output[n*M*L + m*L + l] = sum;
    }
}

void shared_mem_tiling_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {
    
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (L + TILE_L - 1) / TILE_L,
        N
    );
    dim3 block(TILE_M, TILE_L);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "shared_mem_tiling_forward", ([&] {
        shared_mem_tiling_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaDeviceSynchronize();
}

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);
    auto output = torch::zeros({N, M, L}, A.options());
    shared_mem_tiling_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "shared_mem_tiling forward (CUDA)");
}