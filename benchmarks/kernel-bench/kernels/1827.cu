#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 128
#define NUM_STREAMS 4

__global__ void triangular_mm_kernel_coalesced(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N,
                                               int tile_row) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Adjust row to account for tile offset
    row += tile_row;

    // Shared memory for tiles
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    if (row < N && col < N) {
        float sum = 0.f;
        for (int k = 0; k <= row; k += TILE_SIZE) {
            // Load tiles into shared memory
            if (k + threadIdx.x <= row) {
                tile_A[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + (k + threadIdx.x)]);
            } else {
                tile_A[threadIdx.y][threadIdx.x] = 0.f;
            }
            if (k + threadIdx.y <= row) {
                tile_B[threadIdx.y][threadIdx.x] = __ldg(&B[(k + threadIdx.y) * N + col]);
            } else {
                tile_B[threadIdx.y][threadIdx.x] = 0.f;
            }

            __syncthreads();

            // Compute partial product
            for (int n = 0; n < TILE_SIZE; ++n) {
                sum += tile_A[threadIdx.y][n] * tile_B[n][threadIdx.x];
            }

            __syncthreads();
        }
        // Use a mask to avoid branching
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

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);

    // Process matrix in tiles
    for (int tile_row = 0; tile_row < N; tile_row += TILE_SIZE) {
        int current_tile_size = min(TILE_SIZE, N - tile_row);
        dim3 numBlocks((N + threads - 1) / threads,
                      (current_tile_size + threads - 1) / threads);

        // Use stream based on current tile
        int stream_idx = (tile_row / TILE_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel_coalesced<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
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
