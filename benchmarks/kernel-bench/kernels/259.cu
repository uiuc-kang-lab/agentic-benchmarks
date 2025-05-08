#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Declare constant memory for matrix dimensions
__constant__ int d_M;
__constant__ int d_K;
__constant__ int d_N;
__constant__ int d_batch_size;

__global__ void bmm_constant_mem_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C
) {
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    const float* batch_A = A + b * d_M * d_K;
    const float* batch_B = B + b * d_K * d_N;
    
    for (int t = 0; t < (d_K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < d_M && tiledCol < d_K) ? 
            batch_A[row * d_K + tiledCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (tiledRow < d_K && col < d_N) ? 
            batch_B[tiledRow * d_N + col] : 0.0f;
        
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        if (t < (d_K + TILE_SIZE - 1) / TILE_SIZE - 1) {
            __syncthreads();
        }
    }
    
    if (row < d_M && col < d_N) {
        C[b * d_M * d_N + row * d_N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int h_batch_size = A.size(0);
    int h_M = A.size(1);
    int h_K = A.size(2);
    int h_N = B.size(2);

    // Copy dimensions to constant memory
    cudaMemcpyToSymbol(d_batch_size, &h_batch_size, sizeof(int));
    cudaMemcpyToSymbol(d_M, &h_M, sizeof(int));
    cudaMemcpyToSymbol(d_K, &h_K, sizeof(int));
    cudaMemcpyToSymbol(d_N, &h_N, sizeof(int));

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({h_batch_size, h_M, h_N}, options);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((h_N + TILE_SIZE - 1) / TILE_SIZE, 
              (h_M + TILE_SIZE - 1) / TILE_SIZE, 
              h_batch_size);

    bmm_constant_mem_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with constant memory (CUDA)");
}