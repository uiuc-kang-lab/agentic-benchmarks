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
    
    // cuBLAS expects column-major format, but our tensors are row-major
    // To handle this, we compute C = B^T * A^T which is equivalent to (A*B)^T in row-major
    cublasSgemm(handle,
                CUBLAS_OP_N,   // Use B as is (but treat as column-major)
                CUBLAS_OP_N,   // Use A as is (but treat as column-major)
                N, M, K,       // Dimensions are swapped because we're computing (A*B)^T
                &alpha,
                B.data_ptr<float>(), N,  // Leading dimension is N for B
                A.data_ptr<float>(), K,  // Leading dimension is K for A
                &beta,
                C.data_ptr<float>(), N); // Leading dimension is N for C

    // Now C contains the transpose of what we want, so we need to transpose it back
    torch::Tensor C_final = C.transpose(0, 1).contiguous();

    cublasDestroy(handle);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "cuBLAS Matrix Multiplication (CUDA)");
}