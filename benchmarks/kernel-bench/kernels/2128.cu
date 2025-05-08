#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void optimized_triangular_tiling_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {

    __shared__ float Ash[TILE][TILE];
    __shared__ float Bsh[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile <= blockIdx.y; ++tile) {
        const int load_row = blockIdx.y * TILE + threadIdx.y;
        const int load_col = tile * TILE + threadIdx.x;
        if (load_row <= row && load_col <= load_row && load_row < N && load_col < N) {
            Ash[threadIdx.y][threadIdx.x] = __ldg(&A[load_row * N + load_col]);
        } else {
            Ash[threadIdx.y][threadIdx.x] = 0.0f;
        }

        const int b_load_row = tile * TILE + threadIdx.y;
        const int b_load_col = blockIdx.x * TILE + threadIdx.x;
        if (b_load_col <= b_load_row && b_load_row < N && b_load_col < N) {
            Bsh[threadIdx.y][threadIdx.x] = __ldg(&B[b_load_row * N + b_load_col]);
        } else {
            Bsh[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += Ash[threadIdx.y][k] * Bsh[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
    }
}

at::Tensor forward_optimized(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    const dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    const dim3 threads(TILE, TILE);

    optimized_triangular_tiling_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_optimized, "Optimized triangular tiling matmul");
}