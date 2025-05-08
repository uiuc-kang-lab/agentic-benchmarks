#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 32

// Threshold for choosing manual kernel vs. cuBLAS. For small problems, the custom kernel may have lower overhead.
#define MANUAL_THRESHOLD 1000000  // if M * N * K <= 1e6, use manual tiled kernel

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Manual tiled matrix multiplication kernel using shared memory
__global__ void tiled_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Block row and column
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread row and column within the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global row and column indices for C
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tiled_col = tile * TILE_SIZE + tx;
        int tiled_row = tile * TILE_SIZE + ty;

        if (row < M && tiled_col < K)
            As[ty][tx] = A[row * K + tiled_col];
        else
            As[ty][tx] = 0.0f;

        if (tiled_row < K && col < N)
            Bs[ty][tx] = B[tiled_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Persistent cuBLAS handle for larger matrix multiplications
static cublasHandle_t handle = nullptr;

// Unified matrix multiplication function: chooses manual kernel for small matrices and cuBLAS for large matrices
void matrix_multiply_cuda(const torch::Tensor &A,
                            const torch::Tensor &B,
                            torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Determine total operations to decide which path to take
    size_t total_ops = static_cast<size_t>(M) * N * K;

    if (total_ops <= MANUAL_THRESHOLD) {
        // For small problems, launch the manual tiled kernel
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        tiled_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();  // Ensure kernel completes and check for errors if needed
    } else {
        // For larger problems, use the cuBLAS library for optimized performance
        if (handle == nullptr) {
            cublasCreate(&handle);
            // Optionally set math mode; it can use Tensor Cores when available
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Note: cuBLAS expects column-major matrices. The arguments are adjusted accordingly if our data is in row-major order.
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,    // B matrix pointer and leading dimension
                    d_A, K,    // A matrix pointer and leading dimension
                    &beta,
                    d_C, N);   // C matrix pointer and leading dimension
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
    m.def("forward", &forward, "Hybrid matrix multiplication (CUDA) that uses a manual tiled kernel for small matrices and cuBLAS for large matrices");
}
