#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

// Adaptive block size based on hardware capabilities and kernel requirements
#define BLOCK_SIZE 128  // Experiment with different block sizes like 32, 64, 128, 256, 512
#define SMALL_MATRIX_DIM 128

// Macros for input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Static cuBLAS handle for fallback
static cublasHandle_t cublas_handle = nullptr;

// Tiled matrix multiplication kernel with adaptive block size
__global__ void adaptive_blocksize_matmul_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  const int M, const int N, const int K) {
    // Identify thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // Shared memory tiles for A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        if (row < M && t * BLOCK_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (t * BLOCK_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Hybrid matrix multiplication: custom kernel for small matrices, cuBLAS for larger ones
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Use custom kernel for small matrices
    if (M <= SMALL_MATRIX_DIM && N <= SMALL_MATRIX_DIM && K <= SMALL_MATRIX_DIM) {
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        adaptive_blocksize_matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    } else {
        // Use cuBLAS for larger matrices
        if (cublas_handle == nullptr) {
            cublasCreate(&cublas_handle);
            cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Note: cuBLAS assumes column-major order. When using row-major data, swap A and B.
        cublasSgemm(cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    auto options = torch::TensorOptions()
                        .dtype(A.dtype())
                        .device(A.device())
                        .requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);

    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive block size matrix multiplication (CUDA)");
}
