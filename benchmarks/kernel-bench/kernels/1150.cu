#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE = 16;

template <typename scalar_t>
__global__ void module_fn_cuda_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {
    
    __shared__ scalar_t Ash[TILE][TILE];
    __shared__ scalar_t Bsh[TILE][TILE];

    int n = blockIdx.z;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    scalar_t acc = 0;

    for (int k_step = 0; k_step < K; k_step += TILE) {
        int k = k_step + threadIdx.y;
        if (m < M && k < K)
            Ash[threadIdx.x][threadIdx.y] = A[n*M*K + m*K + k];
        else
            Ash[threadIdx.x][threadIdx.y] = 0;

        k = k_step + threadIdx.x;
        if (k < K && l < L)
            Bsh[threadIdx.x][threadIdx.y] = B[k*L + l];
        else
            Bsh[threadIdx.x][threadIdx.y] = 0;

        __syncthreads();

        for (int kk = 0; kk < TILE; ++kk) {
            acc += Ash[threadIdx.x][kk] * Bsh[kk][threadIdx.y];
        }
        __syncthreads();
    }

    if (m < M && l < L)
        output[n*M*L + m*L + l] = acc;
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);

    dim3 threads(TILE, TILE);
    dim3 blocks(
        (M + TILE - 1) / TILE,
        (L + TILE - 1) / TILE,
        N
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        module_fn_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "module_fn forward (CUDA)");
}