#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 64  // Increased tile size for better memory coalescing

#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor")

__global__ void matrixMultiplyKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int M, const int N, const int K) {
    // Use padding to reduce shared memory bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Register blocking for better reuse
    float sum = 0.0f;
    float4 a_reg, b_reg;
    
    // Prefetch first tile
    if (row < M && threadIdx.x < K) {
        As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + threadIdx.x]);
    } else {
        As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    if (threadIdx.y < K && col < N) {
        Bs[threadIdx.y][threadIdx.x] = __ldg(&B[threadIdx.y * N + col]);
    } else {
        Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    #pragma unroll 2
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Double buffering: Load next tile while computing current one
        if (t + 1 < (K + TILE_SIZE - 1) / TILE_SIZE) {
            if (row < M && (t + 1) * TILE_SIZE + threadIdx.x < K) {
                As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + (t + 1) * TILE_SIZE + threadIdx.x]);
            }
            if ((t + 1) * TILE_SIZE + threadIdx.y < K && col < N) {
                Bs[threadIdx.y][threadIdx.x] = __ldg(&B[((t + 1) * TILE_SIZE + threadIdx.y) * N + col]);
            }
        }
        
        // Compute using current tile with aggressive unrolling
        #pragma unroll 16
        for (int k = 0; k < TILE_SIZE; k += 16) {
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    matrixMultiplyKernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA)");
}