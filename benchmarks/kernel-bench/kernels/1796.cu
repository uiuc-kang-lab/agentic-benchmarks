#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized tile size based on typical GPU architectures
#define TILE_SIZE 32
#define BLOCK_ROWS 8
#define VECTORS_PER_THREAD 4

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];
    
    // Calculate base indices
    const int block_row = blockIdx.y * BLOCK_ROWS;
    const int row = block_row + (threadIdx.y * VECTORS_PER_THREAD);
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize accumulator registers
    float sum[VECTORS_PER_THREAD] = {0.0f};
    
    // Early exit if column is out of bounds
    if (col >= N) return;
    
    // Process tiles
    for (int t = col/TILE_SIZE; t <= min(N-1, row + VECTORS_PER_THREAD - 1)/TILE_SIZE; t++) {
        // Collaborative loading of tiles with vectorized loads
        #pragma unroll
        for (int v = 0; v < VECTORS_PER_THREAD; v++) {
            if (row + v < N && (t*TILE_SIZE + threadIdx.x) < N) {
                As[threadIdx.y * VECTORS_PER_THREAD + v][threadIdx.x] = 
                    A[(row + v) * N + (t*TILE_SIZE + threadIdx.x)];
            }
        }
        
        if ((t*TILE_SIZE + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t*TILE_SIZE + threadIdx.y) * N + col];
        }
        
        __syncthreads();
        
        // Compute partial sums using register-level parallelism
        if (col < N) {
            #pragma unroll
            for (int v = 0; v < VECTORS_PER_THREAD; v++) {
                if ((row + v) < N && (row + v) >= col) {
                    float local_sum = 0.0f;
                    #pragma unroll 8
                    for (int k = 0; k < TILE_SIZE; k++) {
                        if ((t*TILE_SIZE + k) >= col && (t*TILE_SIZE + k) <= (row + v)) {
                            local_sum += As[threadIdx.y * VECTORS_PER_THREAD + v][k] * 
                                       Bs[k][threadIdx.x];
                        }
                    }
                    sum[v] += local_sum;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results with vectorized stores
    #pragma unroll
    for (int v = 0; v < VECTORS_PER_THREAD; v++) {
        if (row + v < N && col < N) {
            if (row + v < col) {
                C[(row + v) * N + col] = 0.0f;
            } else {
                C[(row + v) * N + col] = sum[v];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must have same dimensions");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, BLOCK_ROWS);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + (BLOCK_ROWS * VECTORS_PER_THREAD) - 1) / (BLOCK_ROWS * VECTORS_PER_THREAD));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaStreamDestroy(stream);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}