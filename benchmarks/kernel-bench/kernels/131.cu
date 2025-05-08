#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Static cuBLAS handle to avoid recreation costs
static cublasHandle_t handle = nullptr;

// Kernel method to handle cases when Tiling is more effective
__global__ void matmul_kernel_tiled_regtile_optimized(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    //... same as Kernel 1 above ...
}

void optimized_matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    float alpha = 1.0f;
    float beta = 0.0f;

    // Initialize handle only once
    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    
    // Choose strategy based on problem size
    const int TILING_THRESHOLD = 1024; // Hypothetical threshold
    if (M > TILING_THRESHOLD || N > TILING_THRESHOLD || K > TILING_THRESHOLD) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    } else {
        // Define block and grid dimensions
        dim3 blockDim(BLOCK_SIZE / THREAD_TILE_N, BLOCK_SIZE / THREAD_TILE_M);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_kernel_tiled_regtile_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
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

    optimized_matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined tiled and cuBLAS optimized matrix multiplication (CUDA)");
}
