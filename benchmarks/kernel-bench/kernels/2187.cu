#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile sizes and thread block configuration
#define TILE_SIZE 16
#define BLOCK_K 32
#define THREADS_PER_BLOCK 256

__global__ void hybridMatmulKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int K,
                                  int M,
                                  int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    // Grid-stride loop over output elements
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        int i = idx / N;
        int j = idx % N;
        
        float sum = 0.0f;
        
        // Process K dimension in tiles
        for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            int k_start = tile * TILE_SIZE;
            int k_end = min(K, k_start + TILE_SIZE);
            
            // Load tile into shared memory
            if (threadIdx.x < TILE_SIZE) {
                for (int k = k_start; k < k_end; ++k) {
                    if (i < M) As[threadIdx.x][k - k_start] = A[k * M + i];
                    if (j < N) Bs[threadIdx.x][k - k_start] = B[k * N + j];
                }
            }
            __syncthreads();
            
            // Compute partial sum for this tile
            for (int k = 0; k < k_end - k_start; ++k) {
                sum += As[threadIdx.x][k] * Bs[threadIdx.x][k];
            }
            __syncthreads();
        }
        
        if (i < M && j < N) {
            C[i * N + j] = sum;
        }
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

    int total = M * N;
    int num_blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    hybridMatmulKernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        A_ptr, B_ptr, C_ptr, K, M, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid tiled matrix multiplication (CUDA)");
}