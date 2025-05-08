#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Kernel to minimize warp divergence and optimize lower triangular matrix multiplication
__global__ void warp_divergence_optimized_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    // Compute global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Early exit for threads out of bounds
    if (row >= N || col >= N) return;

    // Initialize sum accumulation
    float sum = 0.f;

    // Shared memory tiles for A and B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Number of tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int m = 0; m < numTiles; ++m) {
        // Global k indices
        int tileRow = m * TILE_SIZE + threadIdx.y;
        int tileCol = m * TILE_SIZE + threadIdx.x;

        // Load tiles into shared memory
        sA[threadIdx.y][threadIdx.x] = (tileCol < N && row >= tileCol) ? A[row * N + tileCol] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (tileRow < N && tileRow >= col) ? B[tileRow * N + col] : 0.f;

        // Ensure all threads have loaded their data
        __syncthreads();

        // Reduce warp divergence by iterating in TILE_SIZE steps
        int kStart = max(m * TILE_SIZE, col);
        int kEnd = min((m+1) * TILE_SIZE, N, row+1);

        #pragma unroll
        for (int k = kStart; k < kEnd; ++k) {
            int localK = k - m * TILE_SIZE;
            sum += sA[threadIdx.y][localK] * sB[localK][threadIdx.x];
        }

        // Ensure that computation step is completed
        __syncthreads();
    }

    // Avoid write in upper region which is already resolved by zero initialization initially
    if (row >= col) {
        C[row * N + col] = sum;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    warp_divergence_optimized_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp Divergence Optimized Triangular Matrix Multiplication (CUDA)");
}
