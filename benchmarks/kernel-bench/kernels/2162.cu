#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__device__ __forceinline__ void load_tile(float* shared_tile, const float* global_mat,
                                         int row, int col, int N,
                                         int tile_row, int tile_col) {
    if (row < N && col < N)
        shared_tile[tile_row * TILE_SIZE + tile_col] = global_mat[row * N + col];
    else
        shared_tile[tile_row * TILE_SIZE + tile_col] = 0.0f;
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    const int t_start = col / TILE_SIZE;
    const int t_end = row / TILE_SIZE;

    #pragma unroll 4
    for (int t = t_start; t <= t_end; ++t) {
        load_tile((float*)shA, A, row, t * TILE_SIZE + threadIdx.x, N, threadIdx.y, threadIdx.x);
        load_tile((float*)shB, B, t * TILE_SIZE + threadIdx.y, col, N, threadIdx.y, threadIdx.x);
        __syncthreads();

        const int k_start = max(t * TILE_SIZE, col);
        const int k_end = min((t + 1) * TILE_SIZE, row + 1);
        
        if (k_end - k_start == TILE_SIZE) {
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++)
                sum += shA[threadIdx.y][i] * shB[i][threadIdx.x];
        } else {
            #pragma unroll 8
            for (int k = k_start; k < k_end; k++) {
                const int tile_k = k - t * TILE_SIZE;
                sum += shA[threadIdx.y][tile_k] * shB[tile_k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Requires 2D tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);

    cudaFuncSetCacheConfig(triangular_mm_kernel, cudaFuncCachePreferL1);
    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular mm");
}