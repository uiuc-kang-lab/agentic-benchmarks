#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This function partitions the matrix multiplication across two CUDA streams to overlap execution
// and memory operations. It uses two separate cuBLAS handles, each bound to its stream, to perform
// GEMM on different portions of the output matrix concurrently. This design can help hide latencies
// by overlapping computation with any asynchronous memory operations that might occur.

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    // Dimensions of A: M x K, B: K x N, C: M x N
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Create two CUDA streams for overlapping operations
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Create two cuBLAS handles and assign each a stream
    cublasHandle_t handle1, handle2;
    cublasCreate(&handle1);
    cublasCreate(&handle2);

    cublasSetStream(handle1, stream1);
    cublasSetStream(handle2, stream2);

    // Partition the M dimension into two parts to allow concurrent GEMM operations
    int m1 = M / 2;
    int m2 = M - m1;

    // Launch GEMM on the first partition using stream1, if m1 > 0
    if (m1 > 0) {
        // For cuBLAS, we use the same call as before but with m replaced by m1
        // Note: cuBLAS expects column-major order so we call sgemm with swapped m and n.
        cublasSgemm(handle1,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N,       // number of columns of B and rows of C for this partition
                    m1,      // number of rows in A and C for this partition
                    K,
                    &alpha,
                    d_B,    // B is common for both partitions
                    N,
                    d_A,    // pointer to the beginning of A
                    K,
                    &beta,
                    d_C,    // pointer to the beginning of C
                    N);
    }

    // Launch GEMM on the second partition using stream2, if m2 > 0
    if (m2 > 0) {
        // Offset pointers for the second half of A and C
        float *d_A2 = d_A + m1 * K;
        float *d_C2 = d_C + m1 * N;
        cublasSgemm(handle2,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N,       // number of columns of B and rows of C for this partition
                    m2,      // number of rows in this partition
                    K,
                    &alpha,
                    d_B,    // B remains the same
                    N,
                    d_A2,
                    K,
                    &beta,
                    d_C2,
                    N);
    }

    // Synchronize the streams to ensure all operations are complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Clean up resources
    cublasDestroy(handle1);
    cublasDestroy(handle2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// The forward function allocates the output tensor and calls the custom matrix multiplication.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate the output tensor on CUDA
    torch::Tensor C = torch::zeros({M, N}, A.options());

    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with CUDA streams overlapping computation and memory operations");
}
