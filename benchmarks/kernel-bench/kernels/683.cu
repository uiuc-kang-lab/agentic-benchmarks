#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_CONST_SIZE 8192  // ~64KB limit for H100

template <typename scalar_t>
__constant__ scalar_t const_B[MAX_CONST_SIZE];

template <typename scalar_t>
__global__ void matmul_constant_mem_kernel(const scalar_t* __restrict__ A,
                                         scalar_t* __restrict__ C,
                                         int M, int K, int N) {
    __shared__ scalar_t sA[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    scalar_t sum = 0;

    for (int t = 0; t < K; t += TILE_SIZE) {
        // Load A tile with coalesced accesses
        if (row < M && (t + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Access B from constant memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            if ((t + k) < K)
                sum += sA[threadIdx.y][k] * const_B[(t + k) * N + col];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(B.numel() <= MAX_CONST_SIZE, "B exceeds constant memory capacity");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Copy B to constant memory
    cudaMemcpyToSymbol(const_B, B.data_ptr<scalar_t>(), K*N*sizeof(scalar_t));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE -1)/TILE_SIZE, (M + TILE_SIZE -1)/TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_constant_mem", [&] {
        matmul_constant_mem_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    });

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Constant memory optimized matmul");
}