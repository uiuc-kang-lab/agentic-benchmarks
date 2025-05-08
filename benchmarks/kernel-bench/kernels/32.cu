#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32
#define BLOCK_ROWS 8
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void matmul_optimized_kernel(const float* __restrict__ A, 
                                      const float* __restrict__ B,
                                      float* __restrict__ C, 
                                      const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    // Early exit if out of bounds
    if (row >= N || col >= N) return;

    // Register-level accumulator for better performance
    float C_value = 0.0f;
    
    // Loop over tiles
    #pragma unroll 4
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Prefetch next tile indices
        const int a_idx = row * N + m * TILE_SIZE + tx;
        const int b_idx = (m * TILE_SIZE + ty) * N + col;
        
        // Load tiles using read-only cache
        As[ty][tx] = (m * TILE_SIZE + tx < N) ? __ldg(&A[a_idx]) : 0.0f;
        Bs[ty][tx] = (m * TILE_SIZE + ty < N) ? __ldg(&B[b_idx]) : 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4) {
            C_value += As[ty][k] * Bs[k][tx];
            C_value += As[ty][k+1] * Bs[k+1][tx];
            C_value += As[ty][k+2] * Bs[k+2][tx];
            C_value += As[ty][k+3] * Bs[k+3][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    C[row * N + col] = C_value;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);
    
    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    const int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(TILE_SIZE, BLOCK_ROWS);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, 
                      (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_optimized_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A_data, B_data, C_data, N);
    
    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication (CUDA)");
}