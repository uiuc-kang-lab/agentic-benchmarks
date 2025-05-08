#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16  // Experiment with values: 16, 32, 64, 128, 256, 512
#endif

// Tiled matrix multiplication kernel with tunable BLOCK_SIZE
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles in the K dimension
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int a_index = row * K + t * BLOCK_SIZE + threadIdx.x;
        int b_index = (t * BLOCK_SIZE + threadIdx.y) * N + col;
        
        // Load A tile element with coalesced memory access and reduced branching
        As[threadIdx.y][threadIdx.x] = ((row < M && (t * BLOCK_SIZE + threadIdx.x) < K) ? A[a_index] : 0.0f);

        // Load B tile element with coalesced memory access and reduced branching
        Bs[threadIdx.y][threadIdx.x] = ((col < N && (t * BLOCK_SIZE + threadIdx.y) < K) ? B[b_index] : 0.0f);
        
        __syncthreads();
        
        // Compute the partial product for this tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    // Write the result with boundary check
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Host function called from PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_tiled_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized Matrix Multiplication with tunable block size (CUDA)");
}
