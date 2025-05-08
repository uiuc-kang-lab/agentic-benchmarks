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

    torch::Tensor C = torch::zeros({M, N}, A.options());

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;
    
    // Proper row-major handling through transposed column-major input
    cublasSgemm(handle,
                CUBLAS_OP_T,   // A is treated as KxM (original is row-major MxK)
                CUBLAS_OP_T,   // B is treated as NxK (original is row-major KxN)
                M, N, K,
                &alpha,
                A.data_ptr<float>(), K,
                B.data_ptr<float>(), N,
                &beta,
                C.data_ptr<float>(), M);

    cublasDestroy(handle);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "cuBLAS Matrix Multiplication (CUDA)");
}