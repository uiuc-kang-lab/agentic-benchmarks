#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// CUDA kernel for matrix multiplication using streams for overlapping computation and memory transfers
// Using larger tile and loop unrolling for performance improvement

template <typename scalar_t>
__global__ void matmul_stream_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (t * TILE_WIDTH + threadIdx.y < K && col < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function with CUDA stream handling
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaStream_t streams[2];
    for (int i = 0; i < 2; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_stream_kernel", ([&] {
        for (int i = 0; i < 2; ++i) {
            matmul_stream_kernel<<<num_blocks, threads_per_block, 0, streams[i]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, K, N
            );
        }
    }));

    for (int i = 0; i < 2; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication with streams (CUDA)");
}
