#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void matMulKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int K,
                             int M,
                             int N) {
    __shared__ float s_A[16][16];
    __shared__ float s_B[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        int k = tile * 16 + tx;
        if (k < K && i < M) s_A[ty][tx] = A[k * M + i];
        else s_A[ty][tx] = 0.0f;

        k = tile * 16 + ty;
        if (k < K && j < N) s_B[ty][tx] = B[k * N + j];
        else s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k_step = 0; k_step < 16; ++k_step) {
            sum += s_A[k_step][tx] * s_B[k_step][ty];
        }
        __syncthreads();
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (i < M && j < N && tx == 0) {
        C[i * N + j] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");

    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((M + 15)/16, (N + 15)/16);

    matMulKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matmul with transposed A");
}