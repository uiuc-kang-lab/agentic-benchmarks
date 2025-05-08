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

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n = blockIdx.z;
    const int m = blockIdx.x * TILE + tx;
    const int l = blockIdx.y * TILE + ty;
    
    // Pre-compute base indices to reduce register pressure
    const int base_a = n*M*K + m*K;
    const int base_b = l;

    scalar_t acc = 0;
    
    #pragma unroll
    for (int k_step = 0; k_step < K; k_step += TILE) {
        // Load A tile
        if (m < M && (k_step + ty) < K)
            Ash[tx][ty] = A[base_a + k_step + ty];
        else
            Ash[tx][ty] = 0;

        // Load B tile
        if ((k_step + tx) < K && l < L)
            Bsh[tx][ty] = B[(k_step + tx)*L + base_b];
        else
            Bsh[tx][ty] = 0;

        __syncthreads();

        // Compute dot product for this tile
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += Ash[tx][kk] * Bsh[kk][ty];
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