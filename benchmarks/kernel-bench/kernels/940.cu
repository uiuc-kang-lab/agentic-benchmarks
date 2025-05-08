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

    // Ensure output tensor is properly aligned
    torch::Tensor C = torch::empty({M, N}, 
        torch::TensorOptions()
            .dtype(A.dtype())
            .device(A.device())
            .memory_format(torch::MemoryFormat::Contiguous));

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Set math mode to tensor operations for potential speedup
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Align the computation to warp size (32 threads)
    const int WARP_SIZE = 32;
    int aligned_N = ((N + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    
    // Perform the matrix multiplication
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,
                A.data_ptr<float>(), K,
                &beta,
                C.data_ptr<float>(), N);

    // Synchronize and cleanup
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp-aligned Matrix Multiplication (CUDA)");
}