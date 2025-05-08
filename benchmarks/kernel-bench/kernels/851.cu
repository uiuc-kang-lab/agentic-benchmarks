#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <vector>

#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)
#define NUM_STREAMS 4
#define CHUNK_SIZE 256
#define CUBLAS_THRESHOLD 512

__global__ void stream_matmul_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N,
                                   int chunk_offset) {
    int row0 = blockIdx.y * TILE_DIM + threadIdx.y + chunk_offset;
    int col0 = blockIdx.x * TILE_DIM + threadIdx.x;
    int row1 = row0 + BLOCK_SIZE;
    int col1 = col0 + BLOCK_SIZE;

    float Cvalue00 = 0.0f, Cvalue01 = 0.0f, Cvalue10 = 0.0f, Cvalue11 = 0.0f;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; tile++) {
        int tStart = tile * TILE_DIM;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int aRow = row0 + i * BLOCK_SIZE;
            for (int j = 0; j < 2; j++) {
                int aCol = tStart + threadIdx.x + j * BLOCK_SIZE;
                As[threadIdx.y + i * BLOCK_SIZE][threadIdx.x + j * BLOCK_SIZE] = 
                    (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int bRow = tStart + threadIdx.y + i * BLOCK_SIZE;
            for (int j = 0; j < 2; j++) {
                int bCol = col0 + j * BLOCK_SIZE;
                Bs[threadIdx.y + i * BLOCK_SIZE][threadIdx.x + j * BLOCK_SIZE] = 
                    (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = As[threadIdx.y][k];
            float a_val1 = As[threadIdx.y + BLOCK_SIZE][k];
            float b_val0 = Bs[k][threadIdx.x];
            float b_val1 = Bs[k][threadIdx.x + BLOCK_SIZE];
            Cvalue00 += a_val0 * b_val0;
            Cvalue01 += a_val0 * b_val1;
            Cvalue10 += a_val1 * b_val0;
            Cvalue11 += a_val1 * b_val1;
        }

        __syncthreads();
    }

    if (row0 < M && col0 < N) C[row0 * N + col0] = Cvalue00;
    if (row0 < M && col1 < N) C[row0 * N + col1] = Cvalue01;
    if (row1 < M && col0 < N) C[row1 * N + col0] = Cvalue10;
    if (row1 < M && col1 < N) C[row1 * N + col1] = Cvalue11;
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
        std::vector<cudaStream_t> streams(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
        }

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        int num_chunks = (M + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int stream_idx = chunk % NUM_STREAMS;
            int chunk_offset = chunk * CHUNK_SIZE;
            int chunk_rows = std::min(CHUNK_SIZE, M - chunk_offset);
            
            dim3 grid((N + TILE_DIM - 1) / TILE_DIM, 
                     (chunk_rows + TILE_DIM - 1) / TILE_DIM);

            stream_matmul_kernel<<<grid, block, 0, streams[stream_idx]>>>(
                A.data_ptr<float>(), 
                B.data_ptr<float>(), 
                C.data_ptr<float>(), 
                M, K, N, 
                chunk_offset
            );
        }

        // Synchronize all streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Stream pipelined matrix multiplication (CUDA)");
}