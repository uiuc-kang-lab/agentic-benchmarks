#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 32
#define NUM_STREAMS 4  // Number of concurrent streams

__global__ void bmm_stream_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_offset,
    int M,
    int K,
    int N,
    int batches_per_stream
) {
    int local_batch = blockIdx.z;
    int b = batch_offset + local_batch;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_idx = t * TILE_SIZE + threadIdx.x;
        
        if (row < M && k_idx < K) {
            As[threadIdx.y][threadIdx.x] = batch_A[row * K + k_idx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int k_idy = t * TILE_SIZE + threadIdx.y;
        if (k_idy < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = batch_B[k_idy * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, A.options());

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate batches per stream
    int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batches_per_stream);

    // Launch kernels in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int batch_offset = i * batches_per_stream;
        int current_batches = std::min(batches_per_stream, 
                                     batch_size - batch_offset);
        
        if (current_batches <= 0) break;
        
        // Adjust grid for last stream if needed
        dim3 current_grid = grid;
        current_grid.z = current_batches;
        
        bmm_stream_kernel<<<current_grid, block, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            batch_offset,
            M, K, N,
            current_batches
        );
    }

    // Synchronize all streams
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Stream-based batched matrix multiplication (CUDA)");
}