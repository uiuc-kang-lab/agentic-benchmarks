#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that uses grid-stride loops to cover all elements.
// Unroll the small K-dimension loop to reduce overhead.
__global__ void StridedMatMulKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int K, int N) {
    // Calculate starting row and column for this thread
    int row_start = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute grid stride in y and x dimensions
    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;

    // Loop over rows using grid-stride
    for (int i = row_start; i < M; i += stride_row) {
        // Loop over columns using grid-stride
        for (int j = col_start; j < N; j += stride_col) {
            float sum = 0.0f;
            // Unroll the small K-dimension loop to minimize loop overhead
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// The forward function validates the input and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Choose block dimensions; here 16x16 threads per block
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    StridedMatMulKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(),
                                               B.data_ptr<float>(),
                                               C.data_ptr<float>(),
                                               M, K, N);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication using grid-stride loops with unrolled K loop (CUDA)");
}
