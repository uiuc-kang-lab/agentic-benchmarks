#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using grid-stride loops
__global__ void stride_matmul_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    // Calculate initial row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute strides over rows and cols
    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;

    // Loop over rows and columns using grid-stride
    for (int i = row; i < M; i += stride_row) {
        for (int j = col; j < N; j += stride_col) {
            float sum = 0.0f;
            // Standard dot product over the K dimension
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// PyTorch forward interface wrapping the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto options = torch::TensorOptions()
                       .dtype(A.dtype())
                       .device(A.device());
    
    // Allocate output tensor C
    torch::Tensor C = torch::empty({M, N}, options);

    // Configure block and grid dimensions
    const int blockDimX = 16;
    const int blockDimY = 16;
    dim3 threads(blockDimX, blockDimY);
    dim3 blocks((N + blockDimX - 1) / blockDimX, (M + blockDimY - 1) / blockDimY);

    // Launch the kernel
    stride_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride loop based matrix multiplication (CUDA)");
}
