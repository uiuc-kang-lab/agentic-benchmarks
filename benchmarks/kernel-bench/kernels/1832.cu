#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 128
#define NUM_STREAMS 4
#define UNROLL_FACTOR 4

__global__ void triangular_mm_kernel_unrolled(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int N,
                                              int tile_row) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    row += tile_row;
    
    if (row < N && col < N) {
        float sum = 0.f;
        
        // Handle main loop with unrolling
        int k_start = col;
        int k_end = (row / UNROLL_FACTOR) * UNROLL_FACTOR;
        
        #pragma unroll
        for (int k = k_start; k < k_end; k += UNROLL_FACTOR) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            sum += __ldg(&A[row * N + (k+1)]) * __ldg(&B[(k+1) * N + col]);
            sum += __ldg(&A[row * N + (k+2)]) * __ldg(&B[(k+2) * N + col]);
            sum += __ldg(&A[row * N + (k+3)]) * __ldg(&B[(k+3) * N + col]);
        }
        
        // Handle remaining elements
        for (int k = k_end; k <= row; ++k) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        
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

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);

    for (int tile_row = 0; tile_row < N; tile_row += TILE_SIZE) {
        int current_tile_size = min(TILE_SIZE, N - tile_row);
        dim3 numBlocks((N + threads - 1) / threads,
                      (current_tile_size + threads - 1) / threads);

        int stream_idx = (tile_row / TILE_SIZE) % NUM_STREAMS;
        
        triangular_mm_kernel_unrolled<<<numBlocks, threadsPerBlock, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            tile_row
        );
    }

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