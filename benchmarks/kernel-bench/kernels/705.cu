#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    // Perform matrix multiplication using cuBLAS with stream
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.data_ptr<float>(), N, A.data_ptr<float>(), K, &beta, C.data_ptr<float>(), N);

    // Synchronize the stream
    cudaStreamSynchronize(stream);

    // Destroy CUDA stream
    cudaStreamDestroy(stream);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with CUDA stream (CUDA)");
}