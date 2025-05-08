#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

// Tiling parameters
#define TILE_SIZE 32
#define SMALL_MATRIX_DIM 128

// Macros for input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Static cuBLAS handle for fallback
static cublasHandle_t cublas_handle = nullptr;

// Double-buffered tiled matrix multiplication kernel
// This kernel uses two shared memory buffers to preload the next tile while computing the current tile,
// and synchronizes threads only once per iteration (when switching buffers) for shared memory consistency.
__global__ void double_buffered_matmul_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               const int M, const int N, const int K) {
    // Identify thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Allocate two shared memory buffers for A and B tiles
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int cur = 0;  // current buffer index

    // Preload the first tile into the current buffer
    if (numTiles > 0) {
        int a_col = 0 * TILE_SIZE + tx;
        int b_row = 0 * TILE_SIZE + ty;
        if (row < M && a_col < K)
            As[cur][ty][tx] = A[row * K + a_col];
        else
            As[cur][ty][tx] = 0.0f;

        if (b_row < K && col < N)
            Bs[cur][ty][tx] = B[b_row * N + col];
        else
            Bs[cur][ty][tx] = 0.0f;
    }

    // Ensure the first tile is loaded
    __syncthreads();

    float sum = 0.0f;

    // Loop over all tiles
    for (int t = 0; t < numTiles; t++) {
        int nextTile = t + 1;
        // If there is a next tile, preload it into the alternate buffer
        if (nextTile < numTiles) {
            int next = 1 - cur;
            int a_col = nextTile * TILE_SIZE + tx;
            int b_row = nextTile * TILE_SIZE + ty;
            if (row < M && a_col < K)
                As[next][ty][tx] = A[row * K + a_col];
            else
                As[next][ty][tx] = 0.0f;
            
            if (b_row < K && col < N)
                Bs[next][ty][tx] = B[b_row * N + col];
            else
                Bs[next][ty][tx] = 0.0f;
        }

        // Compute partial product using the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[cur][ty][k] * Bs[cur][k][tx];
        }
        
        // If there is a next tile, synchronize so that the new tile is ready and swap buffers
        if (nextTile < numTiles) {
            __syncthreads();
            cur = 1 - cur;  // swap buffer
        }
    }

    // Write the result if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Hybrid matrix multiplication: custom double-buffered kernel for small matrices, cuBLAS for larger ones
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
        dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
        dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        double_buffered_matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
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
    m.def("forward", &forward, "Double-buffered tiled matrix multiplication (CUDA) with minimal __syncthreads() for shared memory consistency");
}
