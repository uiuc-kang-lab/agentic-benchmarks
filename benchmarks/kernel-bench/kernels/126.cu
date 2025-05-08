#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 32
#define MAX_CONSTANT_K 128

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Constant memory for frequently accessed data
__constant__ float d_constant_A[MAX_CONSTANT_K];
__constant__ int d_matrix_dims[3];  // [M, N, K]

// Static cuBLAS handle
static cublasHandle_t handle = nullptr;

// Custom tiled matrix multiplication kernel with constant memory optimization
__global__ void matmul_kernel_constant(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int M = d_matrix_dims[0];
    const int N = d_matrix_dims[1];
    const int K = d_matrix_dims[2];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load A tile, using constant memory if possible
        if (row < M && tile * TILE_SIZE + tx < K) {
            if (K <= MAX_CONSTANT_K) {
                As[ty][tx] = d_constant_A[row * K + tile * TILE_SIZE + tx];
            } else {
                As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
            }
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B tile
        if (tile * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A,
                         const torch::Tensor &B,
                         torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Copy matrix dimensions to constant memory
    int h_matrix_dims[3] = {M, N, K};
    cudaMemcpyToSymbol(d_matrix_dims, h_matrix_dims, sizeof(int) * 3);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    if (M <= 128 && N <= 128 && K <= 128) {
        // For small matrices, use custom kernel with constant memory optimization
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                      (M + TILE_SIZE - 1) / TILE_SIZE);

        // Copy frequently accessed data to constant memory if it fits
        if (K <= MAX_CONSTANT_K) {
            cudaMemcpyToSymbol(d_constant_A, d_A, K * sizeof(float));
        }

        matmul_kernel_constant<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    } else {
        // For larger matrices, use cuBLAS
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
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
    m.def("forward", &forward, "Constant memory optimized hybrid matrix multiplication (CUDA)");
}