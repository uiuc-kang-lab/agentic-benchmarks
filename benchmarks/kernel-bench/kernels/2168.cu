#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_CONST_SIZE 8192 // 64KB / (4*2) = 8192 float elements per matrix

__constant__ float cA[MAX_CONST_SIZE];
__constant__ float cB[MAX_CONST_SIZE];

__global__ void triangular_mm_kernel(float* __restrict__ C, int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int t_start = col / TILE_SIZE;
    int t_end = row / TILE_SIZE;

    for (int t = t_start; t <= t_end; ++t) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (a_col < N && a_col <= row)
            shA[threadIdx.y][threadIdx.x] = cA[row * N + a_col];
        else
            shA[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && b_row >= col)
            shB[threadIdx.y][threadIdx.x] = cB[b_row * N + col];
        else
            shB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        int k_begin = max(t * TILE_SIZE, col);
        int k_end = min((t + 1) * TILE_SIZE, row + 1);

        #pragma unroll
        for (int k = k_begin; k < k_end; ++k) {
            int local_k = k - t * TILE_SIZE;
            sum += shA[threadIdx.y][local_k] * shB[local_k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A,B must be CUDA tensors");
    const int N = A.size(0);
    TORCH_CHECK(N*N <= MAX_CONST_SIZE, "Matrix dimension exceeds constant memory capacity");

    auto C = torch::empty_like(A);

    cudaMemcpyToSymbol(cA, A.data_ptr<float>(), N*N*sizeof(float));
    cudaMemcpyToSymbol(cB, B.data_ptr<float>(), N*N*sizeof(float));

    dim3 blocks((N + TILE_SIZE-1)/TILE_SIZE, (N + TILE_SIZE-1)/TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    triangular_mm_kernel<<<blocks, threads>>>(C.data_ptr<float>(), N);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular MM with constant memory");
}