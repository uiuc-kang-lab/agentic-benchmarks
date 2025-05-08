#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define NUM_STREAMS 4

__global__ void triangular_mm_kernel_shared(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N,
                                             int tile_row) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    // Adjust row to account for tile offset
    row += tile_row;

    float sum = 0.f;
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load data into shared memory
        if (row < N && m * TILE_SIZE + local_col < N) {
            As[local_row][local_col] = A[row * N + m * TILE_SIZE + local_col];
        } else {
            As[local_row][local_col] = 0.f;
        }

        if (col < N && m * TILE_SIZE + local_row < N) {
            Bs[local_row][local_col] = B[(m * TILE_SIZE + local_row) * N + col];
        } else {
            Bs[local_row][local_col] = 0.f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[local_row][k] * Bs[k][local_col];
        }

        __syncthreads();
    }

    // Use a mask to avoid branching
    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = TILE_SIZE;
    dim3 threadsPerBlock(threads, threads);

    // Process matrix in tiles
    for (int tile_row = 0; tile_row < N; tile_row += TILE_SIZE) {
        int current_tile_size = min(TILE_SIZE, N - tile_row);
        dim3 numBlocks((N + threads - 1) / threads,
                      (current_tile_size + threads - 1) / threads);

        // Use stream based on current tile
        int stream_idx = (tile_row / TILE_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel_shared<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            tile_row
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}