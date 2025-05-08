#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_DIM 32
#define BLOCK_SIZE (BLOCK_DIM * BLOCK_DIM)

__global__ void matmul_kernel(const float* __restrict__ A, 
                             const float* __restrict__ B, 
                             float* __restrict__ C, 
                             const int M, const int N, const int K) {
    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];
    
    // Convert thread ID to 2D coordinates for output matrix
    const int tid = threadIdx.x;
    const int row = blockIdx.y * BLOCK_DIM + (tid / BLOCK_DIM);
    const int col = blockIdx.x * BLOCK_DIM + (tid % BLOCK_DIM);
    
    float sum = 0.0f;
    
    // Iterate over tiles
    for (int t = 0; t < (K + BLOCK_DIM - 1) / BLOCK_DIM; ++t) {
        // Load tiles cooperatively using 1D thread indexing
        const int tileK = t * BLOCK_DIM;
        
        if (row < M && (tileK + (tid % BLOCK_DIM)) < K) {
            As[tid / BLOCK_DIM][tid % BLOCK_DIM] = 
                A[row * K + tileK + (tid % BLOCK_DIM)];
        } else {
            As[tid / BLOCK_DIM][tid % BLOCK_DIM] = 0.0f;
        }
        
        if ((tileK + (tid / BLOCK_DIM)) < K && col < N) {
            Bs[tid / BLOCK_DIM][tid % BLOCK_DIM] = 
                B[(tileK + (tid / BLOCK_DIM)) * N + col];
        } else {
            Bs[tid / BLOCK_DIM][tid % BLOCK_DIM] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_DIM; ++k) {
            sum = __fmaf_rn(As[tid / BLOCK_DIM][k], 
                           Bs[k][tid % BLOCK_DIM], 
                           sum);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    dim3 threads(BLOCK_SIZE);  // 1D thread block
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM,
                (M + BLOCK_DIM - 1) / BLOCK_DIM);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA)");
}