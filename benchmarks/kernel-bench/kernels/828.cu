#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function to validate matrix dimensions
__device__ bool validateDimensions(int M, int K, int N) {
    return (M > 0 && K > 0 && N > 0);
}

// Device function to initialize cuBLAS parameters
__device__ void initializeCublasParams(float* alpha, float* beta) {
    *alpha = 1.0f;
    *beta = 0.0f;
}

// Host function to perform matrix multiplication
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set cuBLAS parameters
    float alpha, beta;
    cudaMemcpyToSymbol(initializeCublasParams, &alpha, sizeof(float));
    cudaMemcpyToSymbol(initializeCublasParams, &beta, sizeof(float));

    // Stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    // Perform matrix multiplication using cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);

    // Cleanup
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA)");
}