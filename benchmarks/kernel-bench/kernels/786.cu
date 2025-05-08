#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int K, const int N) {
    __shared__ float As[BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Collaborative loading into shared memory
        int k = tile * BLOCK_SIZE + threadIdx.x;
        if (row < M && k < K)
            As[threadIdx.x] = A[row * K + k];
        if (col < N && k < K)
            Bs[threadIdx.x] = B[k * N + col];
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE && (tile * BLOCK_SIZE + i) < K; i++) {
            sum += As[i] * Bs[i];
        }
        
        __syncthreads();
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write result
    if (row < M && col < N) {
        if (threadIdx.x % WARP_SIZE == 0) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    dim3 threadsPerBlock(BLOCK_SIZE, 1);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA)");
}