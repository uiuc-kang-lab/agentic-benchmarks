#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__device__ void load_tiles(const float* __restrict__ A, const float* __restrict__ B, 
                          float (&As)[TILE_SIZE][TILE_SIZE], float (&Bs)[TILE_SIZE][TILE_SIZE],
                          int row, int col, int tile, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if ((tile * TILE_SIZE + tx) <= row && row < N) {
        As[ty][tx] = A[row * N + (tile * TILE_SIZE + tx)];
    } else {
        As[ty][tx] = 0.0f;
    }

    if ((tile * TILE_SIZE + ty) < N && col < N) {
        Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
    } else {
        Bs[ty][tx] = 0.0f;
    }

    __syncthreads();
}

__device__ void compute_tile(float (&As)[TILE_SIZE][TILE_SIZE], 
                             float (&Bs)[TILE_SIZE][TILE_SIZE],
                             float &sum, int k_start, int k_end) {
    int tx = threadIdx.x;
    for (int k = k_start; k < k_end; ++k) {
        sum += As[threadIdx.y][k] * Bs[k][tx];
    }
    __syncthreads();
}

__global__ void modular_tril_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int row = by + threadIdx.y;
    int col = bx + threadIdx.x;

    float sum = 0.0f;

    if (row >= col && row < N && col < N) {
        for (int tile = col / TILE_SIZE; tile <= row / TILE_SIZE; ++tile) {
            load_tiles(A, B, As, Bs, row, col, tile, N);
            int k_start = max(col, tile * TILE_SIZE);
            int k_end = min(row, (tile + 1) * TILE_SIZE);
            compute_tile(As, Bs, sum, k_start - tile * TILE_SIZE, k_end - tile * TILE_SIZE);
        }
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    modular_tril_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Modular Triangular Matrix Multiplication (CUDA)");
}
