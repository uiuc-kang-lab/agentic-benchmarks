#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void triangular_mm_kernel_atomic_optimized(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int N) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;

    // Loop over tiles required to compute the (row, col) output element
    for (int sub = 0; sub < gridDim.x; ++sub) {
        int tiledCol = sub * BLOCK_SIZE + threadIdx.x;
        int tiledRow = sub * BLOCK_SIZE + threadIdx.y;

        // Load data into shared memory
        if (row < N && tiledCol < N) {
            tileA[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + tiledCol]);
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiledRow < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledRow * N + col]);
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute sub-block submatrix product within shared memory
        if (row < N && col <= row) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Only store the result if within the lower triangular part
    if (row < N && col <= row) {
        C[row * N + col] = sum;
    }
}

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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaFuncSetCacheConfig(triangular_mm_kernel_atomic_optimized, cudaFuncCachePreferL1);

    triangular_mm_kernel_atomic_optimized<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}