#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_SIZE 16
#define MATRIX_SIZE_THRESHOLD 512

// Optimized kernel to minimize warp divergence
__global__ void WarpDivergenceOptimizedMatmulKernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float cValue = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        // Load tiles with predicated assignments to avoid divergence
        tileA[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;

        __syncthreads();

        // Compute using shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            cValue += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result to C
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

// Forward function that selects the custom kernel for small matrices
// or falls back to cuBLAS for larger ones.
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
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        WarpDivergenceOptimizedMatmulKernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N
        );
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
    m.def("forward", &forward, "Warp divergence optimized matrix multiplication (CUDA)");
}