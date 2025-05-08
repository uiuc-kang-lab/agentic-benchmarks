#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define MATRIX_SIZE_THRESHOLD 512

// Declaration of constant memory for input matrices
__constant__ float A_const[TILE_WIDTH * TILE_WIDTH];
__constant__ float B_const[TILE_WIDTH * TILE_WIDTH];

__global__ void ConstantMemoryMatmulKernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           const int M, const int K, const int N) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float cValue = 0.0f;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;

        if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH) {
            if (row < M && tiledCol < K) {
                A_const[threadIdx.y * TILE_WIDTH + threadIdx.x] = A[row * K + tiledCol];
            }
            if (tiledRow < K && col < N) {
                B_const[threadIdx.y * TILE_WIDTH + threadIdx.x] = B[tiledRow * N + col];
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            cValue += A_const[threadIdx.y * TILE_WIDTH + i] * B_const[i * TILE_WIDTH + threadIdx.x];
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

    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                     (M + TILE_WIDTH - 1) / TILE_WIDTH);
        
        ConstantMemoryMatmulKernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), 
            C.data_ptr<float>(), M, K, N);
    } else {
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
    m.def("forward", &forward, "Constant memory optimized matrix multiplication (CUDA)");
}
