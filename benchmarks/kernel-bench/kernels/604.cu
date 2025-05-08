#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
const int NUM_STREAMS = 4;

// Kernel with improved workload distribution
template <typename scalar_t>
__global__ void matmul_even_workload_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                             scalar_t* __restrict__ C, int M, int K, int N, int row_offset) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = row_offset + blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiledA_col = t * TILE_WIDTH + threadIdx.x;
        int tiledB_row = t * TILE_WIDTH + threadIdx.y;

        if (row < (row_offset + M) && tiledA_col < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledA_col]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && tiledB_row < K)
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledB_row * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i)
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < (row_offset + M) && col < N)
        C[row * N + col] = value;
}

// Host function that splits the workload across multiple CUDA streams

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Dimension mismatch");

    auto C = torch::empty({M, N}, A.options());
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);

    const int chunk_size = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threads(TILE_WIDTH, TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_even_workload_kernel", [&] {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            const int row_start = s * chunk_size;
            const int valid_rows = std::min(chunk_size, static_cast<int>(M - row_start));
            if (valid_rows <= 0) break;

            dim3 blocks((N + TILE_WIDTH-1)/TILE_WIDTH, (valid_rows + TILE_WIDTH-1)/TILE_WIDTH);
            matmul_even_workload_kernel<scalar_t><<<blocks, threads, 0, streams[s]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                valid_rows,
                K,
                N,
                row_start
            );
        }
    });

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Even workload distribution matmul forward");
}
