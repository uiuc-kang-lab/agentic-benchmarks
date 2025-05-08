#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

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
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute C = A*B using tensor cores
    cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), CUDA_R_32F, N,
                A.data_ptr<float>(), CUDA_R_32F, K,
                &beta,
                C.data_ptr<float>(), CUDA_R_32F, N,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasDestroy(handle);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "cuBLAS Tensor Core Matrix Multiplication (CUDA)");
}