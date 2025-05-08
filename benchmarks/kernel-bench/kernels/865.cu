#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define WARP_SIZE 32
#define BLOCK_DIM_X 128
#define BLOCK_DIM_Y 8
#define ELEMENTS_PER_THREAD 4
#define CUBLAS_THRESHOLD 512

template<int ITEMS_PER_THREAD = 4>
__global__ void balanced_matmul_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int M, const int K, const int N) {
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int block_row = blockIdx.y * (BLOCK_DIM_Y * ITEMS_PER_THREAD);
    const int block_col = blockIdx.x * (BLOCK_DIM_X * ITEMS_PER_THREAD);

    __shared__ float As[BLOCK_DIM_Y * ITEMS_PER_THREAD][WARP_SIZE * 2];
    __shared__ float Bs[WARP_SIZE * 2][BLOCK_DIM_X * ITEMS_PER_THREAD];

    float acc[ITEMS_PER_THREAD][ITEMS_PER_THREAD] = {0.0f};

    for (int tile = 0; tile < (K + WARP_SIZE * 2 - 1) / (WARP_SIZE * 2); ++tile) {
        const int tile_idx = tile * (WARP_SIZE * 2);

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            const int row = block_row + threadIdx.y * ITEMS_PER_THREAD + i;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = tile_idx + threadIdx.x + j * WARP_SIZE;
                if (row < M && col < K) {
                    As[threadIdx.y * ITEMS_PER_THREAD + i][threadIdx.x + j * WARP_SIZE] = 
                        A[row * K + col];
                } else {
                    As[threadIdx.y * ITEMS_PER_THREAD + i][threadIdx.x + j * WARP_SIZE] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = tile_idx + threadIdx.y + i * BLOCK_DIM_Y;
            #pragma unroll
            for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
                const int col = block_col + threadIdx.x * ITEMS_PER_THREAD + j;
                if (row < K && col < N) {
                    Bs[threadIdx.y + i * BLOCK_DIM_Y][threadIdx.x * ITEMS_PER_THREAD + j] = 
                        B[row * N + col];
                } else {
                    Bs[threadIdx.y + i * BLOCK_DIM_Y][threadIdx.x * ITEMS_PER_THREAD + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < WARP_SIZE * 2; ++k) {
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                const float a_val = As[threadIdx.y * ITEMS_PER_THREAD + i][k];
                #pragma unroll
                for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
                    acc[i][j] += a_val * Bs[k][threadIdx.x * ITEMS_PER_THREAD + j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        const int row = block_row + threadIdx.y * ITEMS_PER_THREAD + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
                const int col = block_col + threadIdx.x * ITEMS_PER_THREAD + j;
                if (col < N) {
                    C[row * N + col] = acc[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    if (M >= CUBLAS_THRESHOLD && N >= CUBLAS_THRESHOLD && K >= CUBLAS_THRESHOLD) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, M, K, &alpha,
                   B.data_ptr<float>(), N,
                   A.data_ptr<float>(), K,
                   &beta, C.data_ptr<float>(), N);
        cublasDestroy(handle);
    } else {
        dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 grid(
            (N + BLOCK_DIM_X * ELEMENTS_PER_THREAD - 1) / (BLOCK_DIM_X * ELEMENTS_PER_THREAD),
            (M + BLOCK_DIM_Y * ELEMENTS_PER_THREAD - 1) / (BLOCK_DIM_Y * ELEMENTS_PER_THREAD)
        );
        
        balanced_matmul_kernel<ELEMENTS_PER_THREAD><<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N
        );
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Workload balanced matrix multiplication (CUDA)");
}