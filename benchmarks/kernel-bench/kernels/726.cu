#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define TILE_WIDTH 128
#define ELEMENTS_PER_THREAD 4
#define MATRIX_SIZE_THRESHOLD 512

__global__ void WarpOptimizedMatmulKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         const int M, const int K, const int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH/ELEMENTS_PER_THREAD];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH/ELEMENTS_PER_THREAD];
    
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Calculate base indices for this thread
    const int row = blockIdx.y * TILE_WIDTH + (threadIdx.x / (TILE_WIDTH/ELEMENTS_PER_THREAD));
    const int col = blockIdx.x * TILE_WIDTH + (threadIdx.x % (TILE_WIDTH/ELEMENTS_PER_THREAD)) * ELEMENTS_PER_THREAD;
    
    float sum[ELEMENTS_PER_THREAD] = {0.0f};
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load data into shared memory with each thread loading multiple elements
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int shared_idx = threadIdx.x * ELEMENTS_PER_THREAD + i;
            const int k_idx = t * TILE_WIDTH + shared_idx;
            
            if (row < M && k_idx < K) {
                As[threadIdx.x / (TILE_WIDTH/ELEMENTS_PER_THREAD)][shared_idx % (TILE_WIDTH/ELEMENTS_PER_THREAD)] = 
                    A[row * K + k_idx];
            }
            
            if (k_idx < K && col + i < N) {
                Bs[shared_idx / (TILE_WIDTH/ELEMENTS_PER_THREAD)][shared_idx % (TILE_WIDTH/ELEMENTS_PER_THREAD)] = 
                    B[k_idx * N + col + i];
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            const float a_val = As[threadIdx.x / (TILE_WIDTH/ELEMENTS_PER_THREAD)][k];
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                sum[i] += a_val * Bs[k][threadIdx.x % (TILE_WIDTH/ELEMENTS_PER_THREAD) * ELEMENTS_PER_THREAD + i];
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    if (row < M) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            if (col + i < N) {
                C[row * N + col + i] = sum[i];
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
        dim3 threadsPerBlock(THREADS_PER_BLOCK);
        dim3 numBlocks(
            (N + TILE_WIDTH - 1) / TILE_WIDTH,
            (M + TILE_WIDTH - 1) / TILE_WIDTH
        );

        WarpOptimizedMatmulKernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Warp optimized matrix multiplication (CUDA)");
}