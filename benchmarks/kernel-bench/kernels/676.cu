#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_WIDTH 16
#define NUM_STREAMS 4

template <typename scalar_t>
__global__ void matmul_cuda_kernel_streamed(const scalar_t* __restrict__ A,
                                          const scalar_t* __restrict__ B,
                                          scalar_t* __restrict__ C,
                                          int M, int K, int N,
                                          int row_offset) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t value = 0;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threads(TILE_WIDTH, TILE_WIDTH);

    for (int i = 0; i < NUM_STREAMS; i++) {
        int current_chunk_start = i * chunk_size;
        int current_chunk_size = std::min(chunk_size, static_cast<int>(M - current_chunk_start));
        
        if (current_chunk_size <= 0) continue;

        dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                   (current_chunk_size + TILE_WIDTH - 1) / TILE_WIDTH);

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel_streamed", ([&] {
            matmul_cuda_kernel_streamed<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, K, N,
                current_chunk_start
            );
        }));
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA)");
}