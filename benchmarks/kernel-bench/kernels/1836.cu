#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE * 4)

__global__ void triangular_mm_kernel_shared(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            const int N) {
    __shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shB[BLOCK_SIZE][BLOCK_SIZE];

    const int bx = blockIdx.x * BLOCK_SIZE;
    const int by = blockIdx.y * BLOCK_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by + ty;
    const int col = bx + tx;

    float sum = 0.0f;

    // Ensure global memory loads are aligned to 128-bit boundaries
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        const int tile_idx = t * BLOCK_SIZE;

        // Load data into shared memory with aligned access
        if (row < N && (tile_idx + tx) < N) {
            shA[ty][tx] = __ldg(&A[row * N + tile_idx + tx]);
        } else {
            shA[ty][tx] = 0.0f;
        }

        if ((tile_idx + ty) < N && col < N) {
            shB[ty][tx] = __ldg(&B[(tile_idx + ty) * N + col]);
        } else {
            shB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum for this tile
        if (row < N && col < N && row >= col) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                if ((tile_idx + k) <= row && (tile_idx + k) >= col) {
                    sum += shA[ty][k] * shB[k][tx];
                }
            }
        }

        __syncthreads();
    }

    // Write result with coalesced access
    if (row < N && col < N && row >= col) {
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
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel_shared<<<numBlocks, threadsPerBlock>>>(
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