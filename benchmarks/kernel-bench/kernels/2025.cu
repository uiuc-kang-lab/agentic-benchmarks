#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32

__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int M = N * (N + 1) / 2;

    for (int r = idx; r < M; r += stride) {
        float fr = (float)r;
        float tmp = sqrtf(8.0f * fr + 1.0f);
        int row = (int)((tmp - 1.0f) * 0.5f);
        int row_start = (row * (row + 1)) / 2;
        int col = r - row_start;

        float sum = 0.0f;
        int offsetA = row * N;
        int tile_start = (col / TILE_SIZE) * TILE_SIZE;
        int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

        for (int t = tile_start; t < num_tiles && t * TILE_SIZE <= row; ++t) {
            int tile_offset = t * TILE_SIZE;
            load_a_tile(A, row, tile_offset, N, As);
            load_b_tile(B, col, tile_offset, N, Bs);
            __syncthreads();

            int k_start = max(tile_offset, col);
            int k_end = min(tile_offset + TILE_SIZE, row + 1);

            for (int k = k_start; k < k_end; ++k) {
                int k_tile = k - tile_offset;
                sum += As[threadIdx.y][k_tile] * Bs[k_tile][threadIdx.x];
            }
            __syncthreads();
        }

        C[offsetA + col] = sum;
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

    int blockSize = 256;
    int numBlocks = (N * (N + 1) / 2 + blockSize - 1) / blockSize;

    optimized_triangular_mm_kernel<<<numBlocks, blockSize>>>(A.data_ptr<float>(),
                                                             B.data_ptr<float>(),
                                                             C.data_ptr<float>(),
                                                             N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized lower triangular matrix multiplication (CUDA)");
}