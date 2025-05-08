#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void align_memory(float* __restrict__ A, float* __restrict__ B, int M, int K, int N) {
    // Kernel to ensure memory alignment
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * K) {
        A[idx] = __ldg(&A[idx]);
    }
    if (idx < K * N) {
        B[idx] = __ldg(&B[idx]);
    }
}

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
    
    // Align memory accesses
    int numThreads = 256;
    int numBlocksA = (M * K + numThreads - 1) / numThreads;
    int numBlocksB = (K * N + numThreads - 1) / numThreads;
    align_memory<<<numBlocksA, numThreads>>>(A.data_ptr<float>(), B.data_ptr<float>(), M, K, N);

    // cuBLAS expects column-major format, but our tensors are row-major
    cublasSgemm(handle,
                CUBLAS_OP_N,   // Use B as is (but treat as column-major)
                CUBLAS_OP_N,   // Use A as is (but treat as column-major)
                N, M, K,       // Dimensions are swapped because we're computing (A*B)^T
                &alpha,
                B.data_ptr<float>(), N,  // Leading dimension is N for B
                A.data_ptr<float>(), K,  // Leading dimension is K for A
                &beta,
                C.data_ptr<float>(), N); // Leading dimension is N for C

    torch::Tensor C_final = C.transpose(0, 1).contiguous();

    cublasDestroy(handle);
    return C_final;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "cuBLAS Matrix Multiplication with memory alignment (CUDA)");
}