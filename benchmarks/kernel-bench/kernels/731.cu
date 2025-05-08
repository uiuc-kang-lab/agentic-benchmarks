#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define ELEMENTS_PER_THREAD 4
#define MATRIX_SIZE_THRESHOLD 512

__global__ void BalancedMatmulKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int M, const int K, const int N) {
    __shared__ float As[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ float Bs[BLOCK_DIM_X][BLOCK_DIM_X];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    const int col_base = blockIdx.x * (BLOCK_DIM_X * ELEMENTS_PER_THREAD) + threadIdx.x;
    
    float acc[ELEMENTS_PER_THREAD] = {0.0f};
    
    for (int tile = 0; tile < (K + BLOCK_DIM_X - 1) / BLOCK_DIM_X; ++tile) {
        const int tile_idx = tile * BLOCK_DIM_X;
        
        // Load tile of A into shared memory
        if (row < M && (tile_idx + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile_idx + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of B into shared memory
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int col = col_base + i * BLOCK_DIM_X;
            if ((tile_idx + threadIdx.y) < K && col < N) {
                Bs[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X] = 
                    B[(tile_idx + threadIdx.y) * N + col];
            } else {
                Bs[threadIdx.y][threadIdx.x + i * BLOCK_DIM_X] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < BLOCK_DIM_X; ++k) {
            const float a_val = As[threadIdx.y][k];
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                acc[i] += a_val * Bs[k][threadIdx.x + i * BLOCK_DIM_X];
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    if (row < M) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int col = col_base + i * BLOCK_DIM_X;
            if (col < N) {
                C[row * N + col] = acc[i];
            }
        }
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
        dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 gridDim(
            (N + (BLOCK_DIM_X * ELEMENTS_PER_THREAD) - 1) / (BLOCK_DIM_X * ELEMENTS_PER_THREAD),
            (M + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
        );
        
        BalancedMatmulKernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &forward, "Balanced workload matrix multiplication (CUDA)");
}