#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void triangular_mm_optimized(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          const int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int warps_per_block = TILE_SIZE * TILE_SIZE / warpSize;
    int warp_id = thread_id / warpSize;

    int row = (blockIdx.y * TILE_SIZE + warp_id) % N;
    int col = (blockIdx.x * TILE_SIZE + warp_id) / N;
    if (row >= N || col >= N || row < col) return;

    float sum = 0.0f;

    // Start from the appropriate tile
    int t_start = col / TILE_SIZE;
    int t_end = row / TILE_SIZE;

    for (int t = t_start; t <= t_end; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (a_col <= row && a_col < N) {
            shA[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            shA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row >= col && b_row < N) {
            shB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            shB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_optimized<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Optimized Triangular matrix multiplication (CUDA)");
}
