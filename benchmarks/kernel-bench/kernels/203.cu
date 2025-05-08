#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define STREAM_CHUNK 256

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, 
                                   int chunk_rows, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1)/TILE_SIZE; ++t) {
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;

        if (row < chunk_rows && A_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if (B_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < chunk_rows && col < N)
        C[row * N + col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    
    int M = A.size(0), K = A.size(1), N = B.size(1);
    torch::Tensor C = torch::empty({M, N}, A.options());

    float *d_A = A.data_ptr<float>(), *d_B = B.data_ptr<float>(), *d_C = C.data_ptr<float>();
    
    const int num_chunks = (M + STREAM_CHUNK - 1)/STREAM_CHUNK;
    std::vector<cudaStream_t> streams(num_chunks);

    for (int i = 0; i < num_chunks; ++i) {
        cudaStreamCreate(&streams[i]);
        const int chunk_rows = std::min(STREAM_CHUNK, M - i*STREAM_CHUNK);
        
        dim3 grid((N + TILE_SIZE-1)/TILE_SIZE, (chunk_rows + TILE_SIZE-1)/TILE_SIZE);
        dim3 block(TILE_SIZE, TILE_SIZE);
        
        tiled_matmul_kernel<<<grid, block, 0, streams[i]>>>(
            d_A + i*STREAM_CHUNK*K,
            d_B,
            d_C + i*STREAM_CHUNK*N,
            chunk_rows, N, K
        );
    }

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed tiled matrix multiplication");
}
