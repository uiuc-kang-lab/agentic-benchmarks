// This CUDA extension combines an optimized tiled matrix multiplication kernel with an adaptive switch to cuBLAS
// for larger matrix sizes. It uses branchless predicated loads to reduce warp divergence in the custom kernel.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16
#define MATRIX_SIZE_THRESHOLD 512  // Switch point for using cuBLAS for larger matrices

// Optimized tiled matrix multiplication kernel with branchless predicated loads
__global__ void MatmulKernel(const float* __restrict__ A, const float* __restrict__ B, 
                               float* __restrict__ C, int M, int K, int N) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float cValue = 0.0f;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;

        // Branchless load from global memory with predication
        float aElem = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        float bElem = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;
        
        As[threadIdx.y][threadIdx.x] = aElem;
        Bs[threadIdx.y][threadIdx.x] = bElem;
        
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

// The forward function selects the optimal implementation based on matrix size
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());

    // Use custom kernel for small matrices, and cuBLAS for larger matrices
    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
        
        MatmulKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), 
                                            C.data_ptr<float>(), M, K, N);
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    } else {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // Note: cuBLAS assumes column-major storage. Here we adjust by swapping parameters
        cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                          N, M, K, &alpha, 
                                          B.data_ptr<float>(), N, 
                                          A.data_ptr<float>(), K, 
                                          &beta, C.data_ptr<float>(), N);
        TORCH_CHECK(stat == CUBLAS_STATUS_SUCCESS, "cuBLAS sgemm failed");
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive tiled matrix multiplication with cuBLAS fallback");
}
