#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_SIZE 32

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        int lane_id = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        
        // Process tiles using warp-level primitives
        for (int tile = 0; tile < K; tile += WARP_SIZE) {
            float a_val = (tile + lane_id < K) ? A[row * K + tile + lane_id] : 0.0f;
            float b_val = (tile + lane_id < K) ? B[(tile + lane_id) * N + col] : 0.0f;
            
            // Compute partial products within warp
            float partial = a_val * b_val;
            
            // Warp-level reduction using shuffle
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            
            if (lane_id == 0) {
                sum += partial;
            }
        }
        
        // Write result
        if (lane_id == 0) {
            C[row * N + col] = sum;
        }
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    // Ensure inputs are CUDA tensors
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    // Get dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Calculate grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    int M = A.size(0);
    int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA)");
}