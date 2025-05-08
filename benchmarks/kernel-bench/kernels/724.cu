#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define ALIGN_BYTES 16  // For 128-bit alignment

// Helper function to get aligned size
__host__ __device__ inline int get_aligned_size(int size) {
    return (size + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1);
}

__global__ void AlignedMatmulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int M, const int K, const int N) {
    // Ensure shared memory is aligned
    __shared__ __align__(16) float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ __align__(16) float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float cValue = 0.0f;

    // Align K dimension for vectorized loads
    int K_aligned = get_aligned_size(K);
    
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;
        
        // Use vectorized loads where possible
        if (threadIdx.x % 4 == 0 && tiledCol + 3 < K && row < M) {
            float4 aData;
            float4* aPtr = (float4*)&A[row * K + tiledCol];
            aData = *aPtr;
            As[threadIdx.y][threadIdx.x] = aData.x;
            if (threadIdx.x + 1 < TILE_WIDTH) As[threadIdx.y][threadIdx.x + 1] = aData.y;
            if (threadIdx.x + 2 < TILE_WIDTH) As[threadIdx.y][threadIdx.x + 2] = aData.z;
            if (threadIdx.x + 3 < TILE_WIDTH) As[threadIdx.y][threadIdx.x + 3] = aData.w;
        } else {
            // Fall back to scalar loads for boundary cases
            As[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? 
                __ldg(&A[row * K + tiledCol]) : 0.0f;
        }
        
        // Similar vectorized loading for matrix B
        if (threadIdx.y % 4 == 0 && tiledRow + 3 < K && col < N) {
            float4 bData;
            float4* bPtr = (float4*)&B[tiledRow * N + col];
            bData = *bPtr;
            Bs[threadIdx.y][threadIdx.x] = bData.x;
            if (threadIdx.y + 1 < TILE_WIDTH) Bs[threadIdx.y + 1][threadIdx.x] = bData.y;
            if (threadIdx.y + 2 < TILE_WIDTH) Bs[threadIdx.y + 2][threadIdx.x] = bData.z;
            if (threadIdx.y + 3 < TILE_WIDTH) Bs[threadIdx.y + 3][threadIdx.x] = bData.w;
        } else {
            Bs[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? 
                __ldg(&B[tiledRow * N + col]) : 0.0f;
        }
        
        __syncthreads();
        
        // Compute tile multiplication
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            cValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < M && col < N) {
        C[row * N + col] = cValue;
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

    auto C = torch::zeros({M, N}, A.options());

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AlignedMatmulKernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication with aligned memory access (CUDA)");
}