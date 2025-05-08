#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 32

// Macros to check input tensors
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Persistent cuBLAS handle for larger matrices
static cublasHandle_t handle = nullptr;

// Optimized tiled matrix multiplication kernel with aligned global memory loads using __ldg()
__global__ void aligned_tiled_matmul_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              const int M, const int N, const int K) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Block and thread indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Compute the global row and column index for C
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over tiles
    for (int tile = 0; tile < numTiles; ++tile) {
        int aCol = tile * TILE_SIZE + tx;
        int bRow = tile * TILE_SIZE + ty;
        
        // Use __ldg() for read-only global memory loads;
        // It assumes that the underlying data is aligned to 128-bit boundaries if allocated appropriately.
        if (row < M && aCol < K) {
            As[ty][tx] = __ldg(&A[row * K + aCol]);
        } else {
            As[ty][tx] = 0.0f;
        }

        if (bRow < K && col < N) {
            Bs[ty][tx] = __ldg(&B[bRow * N + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute the partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write the computed value to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Hybrid matrix multiplication function: uses the custom kernel for small matrices, cuBLAS for larger ones
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

    // Use the custom kernel when matrices are small
    if (M <= 128 && N <= 128 && K <= 128) {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        aligned_tiled_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();  // Ensure kernel completion (error checking can be added as needed)
    } else {
        // For larger matrices, use the highly optimized cuBLAS routine
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // cuBLAS assumes column-major order. With row-major data, we swap A and B accordingly.
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
    m.def("forward", &forward, "Aligned tiled matrix multiplication (CUDA) with optimized global memory load using __ldg()");
}
