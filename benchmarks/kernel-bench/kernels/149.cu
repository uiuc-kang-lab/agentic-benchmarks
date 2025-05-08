#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

// Constant memory for matrix dimensions
__constant__ int d_M;
__constant__ int d_N;
__constant__ int d_K;

#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor");

__global__ void matrix_multiply_kernel(const float* __restrict__ A, 
                                     const float* __restrict__ B, 
                                     float* __restrict__ C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Using constant memory dimensions
    for (int t = 0; t < (d_K - 1) / TILE_SIZE + 1; ++t) {
        if (row < d_M && t * TILE_SIZE + tx < d_K)
            As[ty][tx] = A[row * d_K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (t * TILE_SIZE + ty < d_K && col < d_N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * d_N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < d_M && col < d_N)
        C[row * d_N + col] = sum;
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Copy dimensions to constant memory
    cudaMemcpyToSymbol(d_M, &M, sizeof(int));
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_K, &K, sizeof(int));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
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