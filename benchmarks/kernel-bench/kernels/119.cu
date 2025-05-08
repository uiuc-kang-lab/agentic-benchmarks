#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define BLOCK_DIM 32
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static cublasHandle_t handle = nullptr;

__global__ void matmul_kernel_2d(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K) {
    
    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_DIM + ty;
    const int col = blockIdx.x * BLOCK_DIM + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + BLOCK_DIM - 1) / BLOCK_DIM; ++tile) {
        const int k_idx = tile * BLOCK_DIM;
        
        // Load A tile with proper boundary checking
        if (row < M && k_idx + tx < K) {
            As[ty][tx] = A[row * K + k_idx + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile with proper boundary checking
        if (k_idx + ty < K && col < N) {
            Bs[ty][tx] = B[(k_idx + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_DIM; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

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

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM,
                (M + BLOCK_DIM - 1) / BLOCK_DIM);

    matmul_kernel_2d<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
}

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
    m.def("forward", &forward, "Optimized 2D matrix multiplication (CUDA)");
}