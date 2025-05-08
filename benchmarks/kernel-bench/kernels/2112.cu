#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel performs triangular matrix multiplication for lower triangular matrices.
// It leverages the __ldg intrinsic to load read-only data efficiently and uses loop unrolling for the inner summation.
// Note: No __syncthreads() are used because the kernel does not employ shared memory, thereby avoiding unnecessary synchronization overhead.

__global__ void triangular_mm_no_sync_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        // For upper triangular part, output is 0
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Compute the dot product for lower triangular element C[row, col] = sum_{k=col}^{row} A[row,k] * B[k,col]
            #pragma unroll
            for (int k = col; k <= row; ++k) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }
            C[row * N + col] = sum;
        }
    }
}

// The C++ interface exposed to PyTorch
at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
                "Matrices must be square and of the same size");

    int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    const int threads = 32;  // 32x32 thread block for optimal warp utilization.
    dim3 block(threads, threads);
    dim3 grid((N + threads - 1) / threads, (N + threads - 1) / threads);

    triangular_mm_no_sync_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication without unnecessary synchronization (CUDA)");
}
