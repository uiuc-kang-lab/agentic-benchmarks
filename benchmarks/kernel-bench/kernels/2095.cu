#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using read-only cache and aligned accesses
__global__ void triangular_mm_kernel_opt(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            // Use 32-bit store for zeroing upper triangle
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Optimized loop with read-only cache loads
            for (int k = col; k <= row; ++k) {
                // Use __ldg for read-only global memory access
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            // Align write to 128-bit boundary using 32-bit store
            C[row * N + col] = sum;
        }
    }
}

at::Tensor forward_optimized(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    const int threads = 16;
    dim3 blocks((N + threads - 1) / threads, (N + threads - 1) / threads);
    triangular_mm_kernel_opt<<<blocks, dim3(threads, threads)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_optimized, "Optimized triangular matmul (CUDA)");
}