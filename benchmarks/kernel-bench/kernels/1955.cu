#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Kernel using dedicated shared memory for frequently reused data
__global__ void dedicated_shared_mem_triangular_mm_kernel(const float* __restrict__ A,
                                                           const float* __restrict__ B,
                                                           float* __restrict__ C,
                                                           int N) {
    // Compute global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Shared memory tiles for A and B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Load tiles of A and B into shared memory with bounds checking
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        int tileRow = m * TILE_SIZE + threadIdx.y;
        int tileCol = m * TILE_SIZE + threadIdx.x;

        if (row < N && tileCol < N && row >= tileCol) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + tileCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tileRow < N && tileRow >= col) {
            sB[threadIdx.y][threadIdx.x] = B[tileRow * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Use pragma unroll for k-dimension
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
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

    // Launch the optimized kernel
    dedicated_shared_mem_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Dedicated Shared Memory Triangular Matrix Multiplication (CUDA)");
}
