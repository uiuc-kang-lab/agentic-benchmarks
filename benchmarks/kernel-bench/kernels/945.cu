#include <torch/extension.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static cublasHandle_t cublas_handle = nullptr;

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());

    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(cublas_handle,
                CUBLAS_OP_N,  // Changed from CUBLAS_OP_T
                CUBLAS_OP_N,  // Changed from CUBLAS_OP_T
                N, M, K,      // Reordered dimensions
                &alpha,
                B.data_ptr<float>(), N,  // Swapped order of matrices
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized cuBLAS matmul with handle reuse");
}