#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
const int NUM_STREAMS = 4;

// Kernel to reduce warp divergence by ensuring uniform control flow

// Device function to load tile from matrix A
__device__ inline float load_A_tile(const float* A, int row, int col, int K, int M) {
    return (row < M && col < K) ? __ldg(&A[row * K + col]) : 0.0f;
}

// Device function to load tile from matrix B
__device__ inline float load_B_tile(const float* B, int row, int col, int N, int K) {
    return (row < K && col < N) ? __ldg(&B[row * N + col]) : 0.0f;
}

// CUDA kernel for matrix multiplication with minimized warp divergence
__global__ void matmul_warp_divergence_reduction_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                                        float* __restrict__ C, int M, int K, int N, int row_offset) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = row_offset + blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiledA_col = t * TILE_WIDTH + threadIdx.x;
        int tiledB_row = t * TILE_WIDTH + threadIdx.y;

        // Load tiles into shared memory using uniform control flow
        sA[threadIdx.y][threadIdx.x] = load_A_tile(A, row, tiledA_col, K, M);
        sB[threadIdx.y][threadIdx.x] = load_B_tile(B, tiledB_row, col, N, K);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i)
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < (row_offset + M) && col < N)
        C[row * N + col] = value;
}

// Host function to invoke the CUDA kernel
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

    for (int s = 0; s < NUM_STREAMS; ++s) {
        const int row_start = s * chunk_size;
        const int valid_rows = std::min(chunk_size, static_cast<int>(M - row_start));
        if (valid_rows <= 0) break;

        dim3 blocks((N + TILE_WIDTH-1)/TILE_WIDTH, (valid_rows + TILE_WIDTH-1)/TILE_WIDTH);
        matmul_warp_divergence_reduction_kernel<<<blocks, threads, 0, streams[s]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            valid_rows,
            K,
            N,
            row_start
        );
    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication with minimized warp divergence");
}