#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 32
#define NUM_STREAMS 4

// Optimized CUDA kernel for symmetric matrix multiplication with atomic operations
__global__ void matmul_atomic_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N,
                                      int chunk_start,
                                      int chunk_size) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty + chunk_start;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    float value = 0.0f;

    for (int i = 0; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        // Coalesced memory access for A
        if (row < chunk_start + chunk_size && i * BLOCK_SIZE + tx < N)
            s_A[ty][tx] = A[row * N + i * BLOCK_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        // Coalesced memory access for B
        if (col < N && i * BLOCK_SIZE + ty < N)
            s_B[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < chunk_start + chunk_size && col < N) {
        atomicAdd(&C[row * N + col], value);
    }
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate chunk size for each stream
    int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Process chunks in parallel using different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_start = i * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - chunk_start);
        
        if (current_chunk_size <= 0) break;

        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_atomic_kernel<<<blocks, threads, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start,
            current_chunk_size
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed Matrix Multiplication with Atomic Operations (CUDA)");
}
