#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32

// Device function to load data from matrix A into shared memory
__device__ __forceinline__ void loadTileA(float (&As)[BLOCK_K][TILE_M],
                                         const float* __restrict__ A,
                                         int k0, int M, int K,
                                         int tid, int totalThreads) {
    #pragma unroll 4
    for (int index = tid; index < BLOCK_K * TILE_M; index += totalThreads) {
        int t = index / TILE_M;
        int m = index % TILE_M;
        int global_k = k0 + t;
        int global_m = blockIdx.x * TILE_M + m;
        As[t][m] = (global_k < K && global_m < M) ? __ldg(&A[global_k * M + global_m]) : 0.0f;
    }
}

// Device function to load data from matrix B into shared memory
__device__ __forceinline__ void loadTileB(float (&Bs)[BLOCK_K][TILE_N],
                                         const float* __restrict__ B,
                                         int k0, int N, int K,
                                         int tid, int totalThreads) {
    #pragma unroll 4
    for (int index = tid; index < BLOCK_K * TILE_N; index += totalThreads) {
        int t = index / TILE_N;
        int n = index % TILE_N;
        int global_k = k0 + t;
        int global_n = blockIdx.y * TILE_N + n;
        Bs[t][n] = (global_k < K && global_n < N) ? __ldg(&B[global_k * N + global_n]) : 0.0f;
    }
}

// Device function to compute partial dot product
__device__ __forceinline__ float computeDotProduct(const float (&As)[BLOCK_K][TILE_M],
                                                  const float (&Bs)[BLOCK_K][TILE_N],
                                                  int tx, int ty) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int t = 0; t < BLOCK_K; t++) {
        sum = __fmaf_rn(As[t][tx], Bs[t][ty], sum);
    }
    return sum;
}

__global__ void modularMatMulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K, int M, int N) {
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int totalThreads = blockDim.x * blockDim.y;
    
    // Global indices
    const int row = blockIdx.x * TILE_M + tx;
    const int col = blockIdx.y * TILE_N + ty;
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Main loop over K dimension
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load tiles from both matrices
        loadTileA(As, A, k0, M, K, tid, totalThreads);
        loadTileB(Bs, B, k0, N, K, tid, totalThreads);
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        sum += computeDotProduct(As, Bs, tx, ty);
        
        __syncthreads();
    }
    
    // Store result
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
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));
    
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    modularMatMulKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular matrix multiplication (CUDA)");
}