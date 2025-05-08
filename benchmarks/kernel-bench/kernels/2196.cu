#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32
#define UNROLL 4

__global__ void tiledSharedMinSyncKernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int K, int M, int N) {
    // Global indices
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;

    // Shared memory tiles
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    // Register for accumulating result
    float sum = 0.0f;

    // Thread ID for coalesced loading
    int tid = threadIdx.y * TILE_M + threadIdx.x;
    int totalThreads = TILE_M * TILE_N;

    // Process K dimension in chunks
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Collaborative loading of A and B tiles into shared memory
        // Each thread loads multiple elements to cover the entire tile
        #pragma unroll
        for (int i = tid; i < BLOCK_K * TILE_M; i += totalThreads) {
            int t = i / TILE_M;
            int m = i % TILE_M;
            int global_k = k0 + t;
            int global_m = blockIdx.x * TILE_M + m;
            
            As[t][m] = (global_k < K && global_m < M) ? A[global_k * M + global_m] : 0.0f;
        }

        #pragma unroll
        for (int i = tid; i < BLOCK_K * TILE_N; i += totalThreads) {
            int t = i / TILE_N;
            int n = i % TILE_N;
            int global_k = k0 + t;
            int global_n = blockIdx.y * TILE_N + n;
            
            Bs[t][n] = (global_k < K && global_n < N) ? B[global_k * N + global_n] : 0.0f;
        }

        // Single sync point after loading shared memory
        __syncthreads();

        // Compute using registers and shared memory
        // No sync needed in this loop as each thread works with its own data
        #pragma unroll
        for (int t = 0; t < BLOCK_K; t += UNROLL) {
            sum += As[t][threadIdx.x] * Bs[t][threadIdx.y];
            if (t + 1 < BLOCK_K) 
                sum += As[t + 1][threadIdx.x] * Bs[t + 1][threadIdx.y];
            if (t + 2 < BLOCK_K)
                sum += As[t + 2][threadIdx.x] * Bs[t + 2][threadIdx.y];
            if (t + 3 < BLOCK_K)
                sum += As[t + 3][threadIdx.x] * Bs[t + 3][threadIdx.y];
        }

        // Single sync point before next iteration to ensure shared memory is ready for reuse
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

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

    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    tiledSharedMinSyncKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using minimal synchronization (CUDA)");
}