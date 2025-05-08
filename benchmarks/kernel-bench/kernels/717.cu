#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define MATRIX_SIZE_THRESHOLD 512  // Threshold to switch between custom kernel and cuBLAS

// Custom kernel for smaller matrices
__global__ void MatmulKernel(const float* __restrict__ A, const float* __restrict__ B, 
                            float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float cValue = 0.0f;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? 
            A[row * K + tiledCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? 
            B[tiledRow * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            cValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
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

    // Choose between custom kernel and cuBLAS based on matrix size
    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        // Use custom kernel for smaller matrices
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                     (M + TILE_WIDTH - 1) / TILE_WIDTH);
        
        MatmulKernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), 
            C.data_ptr<float>(), M, K, N);
    } else {
        // Use cuBLAS for larger matrices
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, M, K, &alpha, 
                   B.data_ptr<float>(), N, 
                   A.data_ptr<float>(), K, 
                   &beta, C.data_ptr<float>(), N);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive matrix multiplication (CUDA)");
}