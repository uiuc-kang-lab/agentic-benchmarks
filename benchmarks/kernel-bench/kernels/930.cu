#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Create dedicated CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    
    // Enable Tensor Core operations
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication using cuBLAS
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
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
    m.def("forward", &matmul_cuda, "Optimized cuBLAS Matrix Multiplication (CUDA)");
}