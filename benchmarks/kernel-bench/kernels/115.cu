/*
Hybrid Matrix Multiplication Extension
This implementation combines a custom tiled CUDA kernel for small matrices and cuBLAS for larger matrices.
For small matrix sizes (e.g. <= 128x128), the custom kernel minimizes launch overhead.
For larger matrices, cuBLAS leverages highly optimized libraries and GPU tensor cores.
*/

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Static cuBLAS handle to avoid recreation overhead
static cublasHandle_t handle = nullptr;

// Custom tiled matrix multiplication kernel
__global__ void matmul_kernel_2d(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Compute row and col for C
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load A tile
        if (row < M && tile * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
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

        // Compute partial dot product using the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Hybrid matrix multiplication: chooses custom kernel for small matrices, cuBLAS for larger ones
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

    // Heuristic: use custom kernel for small matrices, cuBLAS otherwise.
    if (M <= 128 && N <= 128 && K <= 128) {
        // Launch custom tiled kernel
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel_2d<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    } else {
        // Initialize cuBLAS handle if needed
        if (handle == nullptr) {
            cublasCreate(&handle);
            // Optionally, set math mode to use Tensor Cores if available
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Note: cuBLAS assumes column-major order. Here we use arguments in a way that allows using row-major data.
        // We swap A and B pointers so that C = A*B is computed correctly.
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,  // B's leading dimension
                    d_A, K,  // A's leading dimension
                    &beta,
                    d_C, N); // C's leading dimension
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
    m.def("forward", &forward, "Hybrid matrix multiplication (CUDA): custom kernel for small matrices and cuBLAS for large matrices");
}
