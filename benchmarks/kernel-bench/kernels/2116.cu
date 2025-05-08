#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Use u64 arithmetic to avoid narrowing conversions
    const auto block_j = static_cast<uint64_t>(blockIdx.x) * blockDim.x;
    const auto block_i = static_cast<uint64_t>(blockIdx.y) * blockDim.y;

    // Each block processes a tile and checks if fully in lower triangle
    if (block_i < block_j) return; // Skip upper triangle blocks

    const int col = static_cast<int>(block_j + threadIdx.x);
    const int row = static_cast<int>(block_i + threadIdx.y);
    
    if (row >= N || col >= N) return;
    
    if (row >= col) {
        float sum = 0.0f;
        #pragma unroll
        for(int k=col; k<=row; ++k) {
            sum += __ldg(&A[row*N + k]) * __ldg(&B[k*N + col]);
        }
        C[row*N + col] = sum;
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
               "Invalid matrix dimensions");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::zeros_like(A);

    constexpr int threads = 32;
    dim3 block(threads, threads);
    dim3 grid((N + threads -1)/threads, (N + threads -1)/threads);

    triangular_mm_optimized<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "Kernel launch failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized lower-triangular matmul with reduced divergence");
}