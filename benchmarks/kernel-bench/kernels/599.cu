#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32  // Increased tile width for better occupancy
#define NUM_STREAMS 4
#define CHUNKS_PER_STREAM 2  // Process multiple chunks per stream for better overlap

template <typename scalar_t>
__global__ void optimized_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M, const int K, const int N,
    const int row_offset) {
    
    // Increased shared memory for larger tiles
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];
    
    const int row = row_offset + blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    // Use register for accumulation
    scalar_t value = 0;
    
    // Prefetch first tile coordinates
    int curr_tile_idx = 0;
    int tiledA_col = threadIdx.x;
    int tiledB_row = threadIdx.y;
    
    const int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    #pragma unroll 4
    for (int t = 0; t < num_tiles; ++t) {
        // Load current tiles using vectorized loads where possible
        if (row < (row_offset + M) && tiledA_col < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledA_col]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;
            
        if (col < N && tiledB_row < K)
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledB_row * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0;
            
        __syncthreads();
        
        // Compute using registers and unrolled loop
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value = fma(sA[threadIdx.y][i], sB[i][threadIdx.x], value);
        }
        
        __syncthreads();
        
        // Update tile coordinates
        tiledA_col += TILE_WIDTH;
        tiledB_row += TILE_WIDTH;
    }
    
    // Write result using coalesced access
    if (row < (row_offset + M) && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Dimension mismatch");
    
    // Ensure 128-bit alignment
    TORCH_CHECK((reinterpret_cast<uintptr_t>(A.data_ptr()) & 15) == 0, "Input A must be 128-bit aligned");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(B.data_ptr()) & 15) == 0, "Input B must be 128-bit aligned");
    
    auto C = torch::empty({M, N}, A.options());
    
    // Create and store streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    
    const int chunk_size = (M + (NUM_STREAMS * CHUNKS_PER_STREAM) - 1) / (NUM_STREAMS * CHUNKS_PER_STREAM);
    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_matmul_kernel", [&] {
        for (int chunk = 0; chunk < NUM_STREAMS * CHUNKS_PER_STREAM; ++chunk) {
            const int stream_idx = chunk % NUM_STREAMS;
            const int row_start = chunk * chunk_size;
            const int valid_rows = std::min(chunk_size, static_cast<int>(M - row_start));
            
            if (valid_rows <= 0) break;
            
            const dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                            (valid_rows + TILE_WIDTH - 1) / TILE_WIDTH);
                            
            optimized_matmul_kernel<scalar_t><<<blocks, threads, 0, streams[stream_idx]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                valid_rows, K, N,
                row_start
            );
        }
    });
    
    // Cleanup streams
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized multi-stream tiled matrix multiplication");
}