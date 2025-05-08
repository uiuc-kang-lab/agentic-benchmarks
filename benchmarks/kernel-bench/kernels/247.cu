#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ void bmm_warp_uniform_kernel(
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
    
    const int b = blockIdx.z;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Pre-calculate batch offsets
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    float* batch_C = C + b * M * N;
    
    // Pre-calculate validity masks for boundary conditions
    const bool valid_row = row < M;
    const bool valid_col = col < N;
    const bool valid_thread = valid_row && valid_col;
    
    float sum = 0.0f;
    
    // Calculate number of tiles and handle the K dimension uniformly
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 1
    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * TILE_SIZE;
        
        // Load tiles using predicated writes to avoid divergent branches
        const int k_idx = k_base + threadIdx.x;
        const int k_idy = k_base + threadIdx.y;
        
        // Predicated loads for A
        float a_val = 0.0f;
        if (valid_row && k_idx < K) {
            a_val = batch_A[row * K + k_idx];
        }
        As[threadIdx.y][threadIdx.x] = a_val;
        
        // Predicated loads for B
        float b_val = 0.0f;
        if (k_idy < K && valid_col) {
            b_val = batch_B[k_idy * N + col];
        }
        Bs[threadIdx.y][threadIdx.x] = b_val;
        
        __syncthreads();
        
        // Compute partial results - all threads perform the same operations
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Predicated write to global memory
    if (valid_thread) {
        batch_C[row * N + col] = sum;
    }
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

    // Ensure grid dimensions are multiples of warp size where possible
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    bmm_warp_uniform_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with uniform warp execution (CUDA)");
}