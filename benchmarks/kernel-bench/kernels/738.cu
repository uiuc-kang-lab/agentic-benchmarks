#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define MATRIX_SIZE_THRESHOLD 512

__global__ void MinimalSyncMatmulKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int M, const int K, const int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // Register for accumulating results
    float sum = 0.0f;
    
    // Loop over tiles with minimal synchronization
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Each thread loads multiple elements to reduce load instructions
        if (row < M && t * TILE_DIM + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (t * TILE_DIM + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Single sync point after loading shared memory
        __syncthreads();
        
        // Compute using registers for accumulation
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Single sync point before next iteration
        __syncthreads();
    }
    
    // Write result without synchronization (no shared memory access)
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

    auto C = torch::zeros({M, N}, A.options());

    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        dim3 threads(TILE_DIM, BLOCK_ROWS);
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
                 (M + TILE_DIM - 1) / TILE_DIM);

        MinimalSyncMatmulKernel<<<grid, threads>>>(
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
    m.def("forward", &forward, "Minimal sync matrix multiplication (CUDA)");
}