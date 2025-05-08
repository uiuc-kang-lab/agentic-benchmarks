#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Static cuBLAS handle to avoid recreation costs
static cublasHandle_t handle = nullptr;

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Initialize handle only once
    if (handle == nullptr) {
        cublasCreate(&handle);
        // Set math mode to handle Tensor Cores if available
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use CUBLAS_OP_N for both matrices since they are already in the correct layout
    // Perform matrix multiplication C = A * B
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,    // leading dimension of B
                d_A, K,    // leading dimension of A
                &beta,
                d_C, N);   // leading dimension of C
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    // Create output tensor with same options as input
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device())
        .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);

    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication (CUDA)");
}