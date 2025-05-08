#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32
#define NUM_STREAMS 2

__global__ void pipelinedMatMulKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int K, int M, int N,
                                     int chunk_start, int chunk_size) {
    __shared__ float As[2][BLOCK_K][TILE_M];
    __shared__ float Bs[2][BLOCK_K][TILE_N];
    
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;
    
    float sum = 0.0f;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;
    
    int current_buffer = 0;
    
    if (chunk_start < K) {
        for (int index = tid; index < BLOCK_K * TILE_M; index += totalThreads) {
            int t = index / TILE_M;
            int m = index % TILE_M;
            int global_k = chunk_start + t;
            int global_m = blockIdx.x * TILE_M + m;
            As[0][t][m] = (global_k < K && global_m < M) ? __ldg(&A[global_k * M + global_m]) : 0.0f;
        }
        
        for (int index = tid; index < BLOCK_K * TILE_N; index += totalThreads) {
            int t = index / TILE_N;
            int n = index % TILE_N;
            int global_k = chunk_start + t;
            int global_n = blockIdx.y * TILE_N + n;
            Bs[0][t][n] = (global_k < K && global_n < N) ? __ldg(&B[global_k * N + global_n]) : 0.0f;
        }
    }
    
    __syncthreads();
    
    for (int k = chunk_start; k < min(chunk_start + chunk_size, K); k += BLOCK_K) {
        if (k + BLOCK_K < min(chunk_start + chunk_size, K)) {
            #pragma unroll
            for (int index = tid; index < BLOCK_K * TILE_M; index += totalThreads) {
                int t = index / TILE_M;
                int m = index % TILE_M;
                int global_k = k + BLOCK_K + t;
                int global_m = blockIdx.x * TILE_M + m;
                As[1-current_buffer][t][m] = (global_k < K && global_m < M) ? 
                    __ldg(&A[global_k * M + global_m]) : 0.0f;
            }
            
            #pragma unroll
            for (int index = tid; index < BLOCK_K * TILE_N; index += totalThreads) {
                int t = index / TILE_N;
                int n = index % TILE_N;
                int global_k = k + BLOCK_K + t;
                int global_n = blockIdx.y * TILE_N + n;
                Bs[1-current_buffer][t][n] = (global_k < K && global_n < N) ? 
                    __ldg(&B[global_k * N + global_n]) : 0.0f;
            }
        }
        
        #pragma unroll 8
        for (int t = 0; t < BLOCK_K; t++) {
            sum += As[current_buffer][t][threadIdx.x] * Bs[current_buffer][t][threadIdx.y];
        }
        
        current_buffer = 1 - current_buffer;
        __syncthreads();
    }
    
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));
    
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = (K + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_start = i * chunk_size;
        pipelinedMatMulKernel<<<grid, block, 0, streams[i]>>>(
            A_ptr, B_ptr, C_ptr, K, M, N, chunk_start, chunk_size);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined matrix multiplication (CUDA)");
}