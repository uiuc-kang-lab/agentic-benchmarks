#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that computes only the lower triangular part of C = A * B,
// mapping exactly one thread per element in the lower triangular region.
// The total number of threads launched is T = N*(N+1)/2.
__global__ void triangular_mm_kernel_lower_1d(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N,
                                               int total_lower) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_lower) {
        // Map linear index 'idx' to (row, col) in the lower triangular part
        // Use the inverse triangular number formula:
        // row = floor((sqrt(8*t + 1) - 1)/2), and col = t - row*(row+1)/2
        float t = static_cast<float>(idx);
        int row = static_cast<int>((sqrtf(8.0f * t + 1.0f) - 1.0f) * 0.5f);
        int row_start = (row * (row + 1)) / 2;
        int col = idx - row_start;

        float sum = 0.0f;
        // Only indices k from col to row contribute for lower triangular matrices
        #pragma unroll
        for (int k = col; k <= row; ++k) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        C[row * N + col] = sum;
    }
}

// C++ interface exposed to PyTorch.
// We initialize C to zeros so that the upper triangular part remains 0.
// Launch a 1D grid of threads covering only the lower triangular domain (size = N*(N+1)/2).

at::Tensor forward(const at::Tensor &A, const at::Tensor &B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = static_cast<int>(A.size(0));
    // Allocate output tensor initialized to zero so that the upper triangular part is 0
    auto C = torch::zeros_like(A);

    // Total number of elements in the lower triangular part
    int total_lower = (N * (N + 1)) / 2;

    // Launch configuration for the 1D kernel
    int threads = 256;
    int blocks = (total_lower + threads - 1) / threads;

    triangular_mm_kernel_lower_1d<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        total_lower
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "1D lower-triangular matmul with efficient thread mapping (CUDA)");
}
