#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void shared_memory_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Calculate number of tiles needed
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile into shared memory
        int tileRow = threadIdx.y;
        int tileCol = threadIdx.x;
        int globalRow = row;
        int globalCol = t * TILE_SIZE + tileCol;
        
        if (globalRow < N && globalCol < N && globalRow >= globalCol) {
            As[tileRow][tileCol] = A[globalRow * N + globalCol];
        } else {
            As[tileRow][tileCol] = 0.0f;
        }
        
        globalRow = t * TILE_SIZE + tileRow;
        globalCol = col;
        if (globalRow < N && globalCol < N && globalRow >= globalCol) {
            Bs[tileRow][tileCol] = B[globalRow * N + globalCol];
        } else {
            Bs[tileRow][tileCol] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        if (row < N && col < N) {
            if (row >= col) {
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; k++) {
                    int globalK = t * TILE_SIZE + k;
                    if (globalK <= row && globalK >= col) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        } else {
            C[row * N + col] = sum;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);
    
    shared_memory_tiled_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Shared memory tiled triangular matmul (CUDA)");
}