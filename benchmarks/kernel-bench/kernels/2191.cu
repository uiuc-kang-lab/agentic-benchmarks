#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// Tile dimensions
#define TILE_M 16  // Tile size in the M dimension (output row)
#define TILE_N 16  // Tile size in the N dimension (output column)
#define BLOCK_K 32 // Chunk size along the K dimension

__global__ void streamedTiledSharedKernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int K,
                                           int M,
                                           int N) {
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;

    float value = 0.0f;
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int tid = threadIdx.y * TILE_M + threadIdx.x;
        int numThreads = TILE_M * TILE_N;

        for (int idx = tid; idx < BLOCK_K * TILE_M; idx += numThreads) {
            int t = idx / TILE_M;
            int m_idx = idx % TILE_M;
            int global_i = blockIdx.x * TILE_M + m_idx;
            int global_k = k0 + t;
            if (global_i < M && global_k < K)
                As[t][m_idx] = A[global_k * M + global_i];
            else
                As[t][m_idx] = 0.0f;
        }

        for (int idx = tid; idx < BLOCK_K * TILE_N; idx += numThreads) {
            int t = idx / TILE_N;
            int n_idx = idx % TILE_N;
            int global_j = blockIdx.y * TILE_N + n_idx;
            int global_k = k0 + t;
            if (global_j < N && global_k < K)
                Bs[t][n_idx] = B[global_k * N + global_j];
            else
                Bs[t][n_idx] = 0.0f;
        }

        __syncthreads();

        for (int t = 0; t < BLOCK_K; ++t) {
            value += As[t][threadIdx.x] * Bs[t][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function using CUDA streams

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    for (int i = 0; i < num_streams; ++i) {
        streamedTiledSharedKernel<<<grid, block, 0, streams[i]>>>(A_ptr, B_ptr, C_ptr, K, M, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaStreamSynchronize(streams[i]);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using streamed tiled shared memory (CUDA)");
}
