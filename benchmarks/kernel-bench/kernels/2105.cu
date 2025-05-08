#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
__device__ inline bool thread_in_bounds(int row, int col, int N) {
    return (row < N && col < N);
}

__device__ inline bool is_lower_triangular(int row, int col) {
    return row >= col;
}

__device__ inline float compute_element(
    const float* __restrict__ A,
    const float* __restrict__ B,
    int row,
    int col,
    int N
) {
    float sum = 0.0f;
    #pragma unroll
    for(int k = col; k <= row; ++k) {
        sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
    }
    return sum;
}
} // anonymous namespace

__global__ void optimized_tril_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(thread_in_bounds(row, col, N)) {
        if(is_lower_triangular(row, col)) {
            C[row * N + col] = compute_element(A, B, row, col, N);
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
                "Invalid matrix dimensions");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    constexpr int threads = 32;
    const dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);
    
    optimized_tril_mm_kernel<<<blocks, dim3(threads, threads)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}