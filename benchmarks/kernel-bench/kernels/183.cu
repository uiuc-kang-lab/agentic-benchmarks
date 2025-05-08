#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    // Get the dimensions of the matrices
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Get the pointers to the data
    const float *d_A = A.data_ptr<float>();
    const float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrix_multiply_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    // Get the dimensions of the matrices
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create the output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Perform the matrix multiplication
    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with memory coalescing (CUDA)");
}