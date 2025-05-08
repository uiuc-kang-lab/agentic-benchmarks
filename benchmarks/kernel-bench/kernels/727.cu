#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define MATRIX_SIZE_THRESHOLD 512

__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void WarpOptimizedMatmulKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         const int M, const int K, const int N) {
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.y * (BLOCK_SIZE/WARP_SIZE) + warpId;
    
    // Each thread handles multiple columns
    for (int col = blockIdx.x * WARP_SIZE + laneId; col < N; col += gridDim.x * WARP_SIZE) {
        float sum = 0.0f;
        
        // Compute partial products
        for (int k = 0; k < K; k++) {
            if (row < M) {
                sum += A[row * K + k] * B[k * N + col];
            }
        }
        
        // Only write result if within bounds
        if (row < M && col < N) {
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

    auto C = torch::zeros({M, N}, A.options());

    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        // Calculate grid dimensions for warp-based kernel
        dim3 blockDim(BLOCK_SIZE);  // Multiple warps per block
        dim3 gridDim(
            (N + WARP_SIZE - 1) / WARP_SIZE,  // Columns
            (M + (BLOCK_SIZE/WARP_SIZE) - 1) / (BLOCK_SIZE/WARP_SIZE)  // Rows
        );

        WarpOptimizedMatmulKernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N
        );
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
    m.def("forward", &forward, "Warp-optimized matrix multiplication (CUDA)");
}