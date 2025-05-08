#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32
#define TILE_SIZE 16

__global__ void matmul_warp_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const unsigned int FULL_MASK = 0xffffffff;
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    for (int i = 0; i < K; i += TILE_SIZE) {
        if (row < M && (i + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + i + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && (i + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        float local_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            local_sum = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], local_sum);
        }
        sum += local_sum;
        
        // Warp-level reduction
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        if (lane_id == 0) {  // Only first thread in warp writes result
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
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_warp_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Matrix Multiplication (CUDA)");
}