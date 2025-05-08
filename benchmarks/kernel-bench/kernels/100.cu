#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define constant memory for matrix B
__constant__ float const_B[1024 * 1024]; // Assuming maximum size of B

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
    float *d_A = A.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Copy matrix B to constant memory
    cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), K * N * sizeof(float));

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform the matrix multiplication using CUBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, const_B, N, d_A, K, &beta, d_C, N);

    // Destroy the CUBLAS handle
    cublasDestroy(handle);
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
    m.def("forward", &forward, "Matrix multiplication with constant memory optimization (CUDA)");
}