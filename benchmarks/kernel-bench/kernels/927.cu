#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void transpose_kernel(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < width && y < height) {
        out[y + x * height] = __ldg(&in[x + y * width]);
    }
}

void transpose(const float* in, float* out, int width, int height, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    transpose_kernel<<<blocks, threads, 0, stream>>>(in, out, width, height);
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    // Perform GEMM directly using cuBLAS column-major format
    // Note: cuBLAS expects matrices in column-major format, while PyTorch uses row-major
    // So we compute C = (B^T * A^T)^T which is equivalent to C = A * B in row-major
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B.data_ptr<float>(), N,    // B is treated as transposed
                A.data_ptr<float>(), K,    // A is treated as transposed
                &beta,
                C.data_ptr<float>(), N);

    cublasDestroy(handle);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "cuBLAS Matrix Multiplication with memory access optimization (CUDA)");
}
