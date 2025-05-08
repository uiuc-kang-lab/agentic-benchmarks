#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 32

// CUDA kernel for computing C = A.T * B using optimized shared memory synchronization with atomic operations.
__global__ void matMulAtomicOptimizedKernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int K, int M, int N) {
    // Use registers for thread-local computation
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;
    
    // Declare sum in register for faster access
    float sum = 0.0f;
    
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];  // +1 padding

    // Calculate number of tiles needed
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Main computation loop
    for (int t = 0; t < numTiles; t++) {
        // Load data into shared memory with coalesced access pattern
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        
        // Coalesced global memory access
        if (aRow < M && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[aCol * M + aRow];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (bRow < K && bCol < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product using registers
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

// The forward function exposed via PyBind11.
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

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    matMulAtomicOptimizedKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using optimized shared memory tiling with atomic operations");
}