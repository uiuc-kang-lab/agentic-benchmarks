#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_multi_stream_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int active_N,
    const int K,
    const int col_start) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];

    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col_local = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = col_start + col_local;
    
    scalar_t sum = 0;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int a_k = tile * TILE_SIZE + threadIdx.y;
        bool a_valid = (a_k < K) && (row < M);
        A_shared[threadIdx.y][threadIdx.x] = a_valid ? A[a_k * M + row] : 0;

        int b_k = tile * TILE_SIZE + threadIdx.x;
        bool b_valid = (b_k < K) && (col < N);
        B_shared[threadIdx.x][threadIdx.y] = b_valid ? B[col * K + b_k] : 0;

        __syncthreads();

        if (row < M && col_local < active_N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += A_shared[k][threadIdx.x] * B_shared[k][threadIdx.y];
            }
        }
        __syncthreads();
    }

    if (row < M && col < N && col_local < active_N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    auto C = torch::empty({M, N}, A.options());

    const int TILE_SIZE = 32;
    const int num_streams = 4;
    const int chunk_size = (N + num_streams - 1) / num_streams;

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (int s = 0; s < num_streams; ++s) {
        int col_start = s * chunk_size;
        int col_end = std::min(col_start + chunk_size, N);
        int active_N = col_end - col_start;
        
        if (active_N <= 0) continue;

        dim3 blocks(
            (M + TILE_SIZE - 1) / TILE_SIZE,
            (active_N + TILE_SIZE - 1) / TILE_SIZE
        );
        blocks.x = min(blocks.x, 65535); // Limit to maximum grid size
        blocks.y = min(blocks.y, 65535); // Limit to maximum grid size

        AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_multi_stream", ([&] {
            matmul_transpose_multi_stream_kernel<scalar_t>
                <<<blocks, dim3(TILE_SIZE, TILE_SIZE), 0, streams[s]>>>(
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    C.data_ptr<scalar_t>(),
                    M, N, active_N, K, col_start
                );
        }));
    }

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Multi-stream matrix multiplication with transposed inputs");
}