#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Optimized tile size for shared memory/occupancy balance

__global__ void bmm_optimized_kernel(
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
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // Precompute batch pointers for faster address calculations
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative loading with combined bounds checking
        int loadA_col = t * TILE_SIZE + threadIdx.x;
        int loadB_row = t * TILE_SIZE + threadIdx.y;
        
        // As tile loading with thread.y frozen for better coalescing
        if (row < M && loadA_col < K) {
            As[threadIdx.y][threadIdx.x] = batch_A[row * K + loadA_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Bs tile loading with thread.x frozen for better coalescing
        if (loadB_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = batch_B[loadB_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        
        // Unrolled inner loop for reduced overhead
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
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

    auto C = torch::zeros({batch_size, M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );

    bmm_optimized_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Optimized batched matrix multiplication (CUDA)");
}