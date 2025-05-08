#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    // 1D thread indexing with coalesced memory access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Loop unrolling for better instruction-level parallelism
            #pragma unroll(4)
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Optimized 1D grid configuration for better occupancy
    const int threads_per_block = 256;
    const int total_threads = N * N;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    triangular_mm_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication");
}