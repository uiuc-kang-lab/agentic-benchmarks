#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for batched matrix multiplication using stride loops
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
// Tile size for shared memory optimization
#define TILE_SIZE 16

__global__ void bmm_kernel_stride(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;  // batch index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float sum = 0.0f;
    
    // Check if this thread's indices are within bounds
    if (bz < batch_size && row < M && col < N) {
        // Loop over tiles
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
            // Load tile from A into shared memory
            if (t * TILE_SIZE + tx < K && row < M) {
                As[ty][tx] = A[bz * M * K + row * K + t * TILE_SIZE + tx];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            // Load tile from B into shared memory
            if (t * TILE_SIZE + ty < K && col < N) {
                Bs[ty][tx] = B[bz * K * N + (t * TILE_SIZE + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product for this tile
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[ty][k] * Bs[k][tx];
            }
            
            __syncthreads();
        }
        
        // Write result to global memory
        if (row < M && col < N) {
            C[bz * M * N + row * N + col] = sum;
        }
    }
}

// Torch binding function
torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    int total = batch_size * M * N;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bmm_kernel_stride<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication (CUDA) with stride loops");
}
