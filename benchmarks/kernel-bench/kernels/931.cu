#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BATCH_SIZE 4  // Tune this based on matrix size
#define CHUNK_SIZE 1024

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Enable tensor cores if available
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Calculate optimal chunk sizes
    int m_chunks = (M + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    // Prepare for batched operation
    long long int strideA = K * CHUNK_SIZE;
    long long int strideB = 0;  // B remains the same for all chunks
    long long int strideC = N * CHUNK_SIZE;

    for (int chunk = 0; chunk < m_chunks; chunk++) {
        int current_M = (chunk == m_chunks - 1) ? (M - chunk * CHUNK_SIZE) : CHUNK_SIZE;
        
        float* A_ptr = A.data_ptr<float>() + chunk * CHUNK_SIZE * K;
        float* B_ptr = B.data_ptr<float>();
        float* C_ptr = C.data_ptr<float>() + chunk * CHUNK_SIZE * N;

        cublasGemmStridedBatchedEx(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  N, current_M, K,
                                  &alpha,
                                  B_ptr, CUDA_R_32F, N, strideB,
                                  A_ptr, CUDA_R_32F, K, strideA,
                                  &beta,
                                  C_ptr, CUDA_R_32F, N, strideC,
                                  1,  // batch count
                                  CUDA_R_32F,
                                  CUBLAS_GEMM_DEFAULT);
    }

    cublasDestroy(handle);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Batched Matrix Multiplication (CUDA)");
}