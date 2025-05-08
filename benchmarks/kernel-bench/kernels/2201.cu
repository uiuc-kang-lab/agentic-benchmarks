#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32

__constant__ int cK, cM, cN;

__global__ void tiledSharedConstKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C) {
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;

    float sum = 0.0f;

    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    int tid = threadIdx.y * TILE_M + threadIdx.x;
    int totalThreads = TILE_M * TILE_N;

    for (int k0 = 0; k0 < cK; k0 += BLOCK_K) {
        for (int index = tid; index < BLOCK_K * TILE_M; index += totalThreads) {
            int t = index / TILE_M;
            int i = index % TILE_M;
            int global_i = blockIdx.x * TILE_M + i;
            int global_k = k0 + t;
            As[t][i] = (global_i < cM && global_k < cK) ? A[global_k * cM + global_i] : 0.0f;
        }

        for (int index = tid; index < BLOCK_K * TILE_N; index += totalThreads) {
            int t = index / TILE_N;
            int j = index % TILE_N;
            int global_j = blockIdx.y * TILE_N + j;
            int global_k = k0 + t;
            Bs[t][j] = (global_j < cN && global_k < cK) ? B[global_k * cN + global_j] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < BLOCK_K; t += 4) {
            sum += As[t][threadIdx.x] * Bs[t][threadIdx.y]
                 + As[t+1][threadIdx.x] * Bs[t+1][threadIdx.y]
                 + As[t+2][threadIdx.x] * Bs[t+2][threadIdx.y]
                 + As[t+3][threadIdx.x] * Bs[t+3][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < cM && col < cN) {
        C[row * cN + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "A and B must have same K");
    int N = B.size(1);

    cudaMemcpyToSymbol(cK, &K, sizeof(int));
    cudaMemcpyToSymbol(cM, &M, sizeof(int));
    cudaMemcpyToSymbol(cN, &N, sizeof(int));

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    tiledSharedConstKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "C = A.T @ B with constant memory optimization");
}
