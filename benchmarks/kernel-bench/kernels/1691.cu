#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using stride loops to cover matrix elements beyond the initial grid
__global__ void triangular_mm_stride_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int N) {
    // Calculate initial row and column indices for this thread
    int start_row = blockIdx.y * blockDim.y + threadIdx.y;
    int start_col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the strides in each dimension
    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;

    // Loop over the matrix with strides to cover all elements
    for (int row = start_row; row < N; row += stride_row) {
        for (int col = start_col; col < N; col += stride_col) {
            if (row < col) {
                C[row * N + col] = 0.0f;
            } else {
                float sum = 0.0f;
                // Accumulate only over the range contributing to the lower triangular part
                for (int k = col; k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// PyTorch interface
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Use 32x32 threads per block. Stride loops ensure that the entire matrix is processed
    dim3 threads(32, 32);
    // The grid size can be smaller than the matrix dimensions since the stride loops cover
    // remaining rows and columns
    dim3 blocks((N + 31) / 32, (N + 31) / 32);

    triangular_mm_stride_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication with stride loops (CUDA)");
}
