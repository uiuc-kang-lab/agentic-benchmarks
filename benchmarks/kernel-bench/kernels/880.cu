#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_DIM 32
#define ELEMENTS_PER_THREAD 4

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int K, const int N) {
    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int bid_x = blockIdx.x * BLOCK_DIM;
    const int bid_y = blockIdx.y * BLOCK_DIM * ELEMENTS_PER_THREAD;
    
    float acc[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD] = {0.0f};
    
    for (int tile = 0; tile < (K + BLOCK_DIM - 1) / BLOCK_DIM; ++tile) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
                const int row = bid_y + tid_y + i * BLOCK_DIM;
                const int col = tile * BLOCK_DIM + tid_x + j * BLOCK_DIM;
                if (row < M && col < K) {
                    As[tid_y + i * BLOCK_DIM][tid_x + j * BLOCK_DIM] = A[row * K + col];
                } else {
                    As[tid_y + i * BLOCK_DIM][tid_x + j * BLOCK_DIM] = 0.0f;
                }
                
                const int b_row = tile * BLOCK_DIM + tid_y + i * BLOCK_DIM;
                const int b_col = bid_x + tid_x + j * BLOCK_DIM;
                if (b_row < K && b_col < N) {
                    Bs[tid_y + i * BLOCK_DIM][tid_x + j * BLOCK_DIM] = B[b_row * N + b_col];
                } else {
                    Bs[tid_y + i * BLOCK_DIM][tid_x + j * BLOCK_DIM] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
                for (int k = 0; k < BLOCK_DIM; ++k) {
                    acc[i][j] += As[tid_y + i * BLOCK_DIM][k] * 
                                 Bs[k][tid_x + j * BLOCK_DIM];
                }
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int row = bid_y + tid_y + i * BLOCK_DIM;
            const int col = bid_x + tid_x + j * BLOCK_DIM;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(
        (N + BLOCK_DIM * ELEMENTS_PER_THREAD - 1) / (BLOCK_DIM * ELEMENTS_PER_THREAD),
        (M + BLOCK_DIM * ELEMENTS_PER_THREAD - 1) / (BLOCK_DIM * ELEMENTS_PER_THREAD)
    );
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA)");
}