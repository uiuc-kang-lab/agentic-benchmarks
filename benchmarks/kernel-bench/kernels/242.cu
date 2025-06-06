#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy

// Declare constant memory for frequently accessed data
__constant__ float const_A[1024 * 1024];  // Adjust size as needed
__constant__ float const_B[1024 * 1024];  // Adjust size as needed

__global__ void bmm_constant_memory_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Base pointers for current batch
    const float* batch_A = const_A + b * M * K;
    const float* batch_B = const_B + b * K * N;
    
    // Process tiles with minimal synchronization
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_idx = t * TILE_SIZE + threadIdx.x;
        int k_idy = t * TILE_SIZE + threadIdx.y;
        
        // Load tiles into shared memory
        if (row < M && k_idx < K) {
            As[threadIdx.y][threadIdx.x] = batch_A[row * K + k_idx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (k_idy < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = batch_B[k_idy * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Single sync after loading shared memory
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Single sync before next iteration
        __syncthreads();
    }
    
    // Write result if within bounds (no sync needed)
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

void copy_to_constant_memory(torch::Tensor A, torch::Tensor B) {
    cudaMemcpyToSymbol(const_A, A.data_ptr<float>(), A.numel() * sizeof(float));
    cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), B.numel() * sizeof(float));
}

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

    // Copy data to constant memory
    copy_to_constant_memory(A, B);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE, 
              batch_size);

    bmm_constant_memory_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with constant memory (CUDA)");
}