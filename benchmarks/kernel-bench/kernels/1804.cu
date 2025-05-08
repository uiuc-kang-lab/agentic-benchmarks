#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy
#define BLOCK_ROWS 8  // Reduced block rows for better register usage

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][BLOCK_ROWS];
    __shared__ float Bs[BLOCK_ROWS][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
            return;
        }
        
        // Process tiles with shared memory
        for (int t = col/BLOCK_ROWS; t <= row/BLOCK_ROWS; t++) {
            // Load tile strips into shared memory
            #pragma unroll 4
            for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
                if (row < N && (t*BLOCK_ROWS + i) <= row) {
                    As[ty + i][tx % BLOCK_ROWS] = A[row * N + (t*BLOCK_ROWS + i)];
                }
                if ((t*BLOCK_ROWS + i) < N && col < N) {
                    Bs[tx % BLOCK_ROWS][ty + i] = B[(t*BLOCK_ROWS + i) * N + col];
                }
            }
            
            __syncthreads();
            
            // Compute partial sum with unrolled inner loop
            if (row >= col) {
                int k_start = max(t*BLOCK_ROWS, col);
                int k_end = min((t+1)*BLOCK_ROWS, row + 1);
                
                #pragma unroll 4
                for (int k = k_start; k < k_end - 3; k += 4) {
                    sum += As[ty][k % BLOCK_ROWS] * Bs[k % BLOCK_ROWS][tx];
                    sum += As[ty][(k+1) % BLOCK_ROWS] * Bs[(k+1) % BLOCK_ROWS][tx];
                    sum += As[ty][(k+2) % BLOCK_ROWS] * Bs[(k+2) % BLOCK_ROWS][tx];
                    sum += As[ty][(k+3) % BLOCK_ROWS] * Bs[(k+3) % BLOCK_ROWS][tx];
                }
                
                // Handle remaining elements
                for (int k = k_start + ((k_end - k_start)/4)*4; k < k_end; k++) {
                    sum += As[ty][k % BLOCK_ROWS] * Bs[k % BLOCK_ROWS][tx];
                }
            }
            
            __syncthreads();
        }
        
        if (row >= col) {
            C[row * N + col] = sum;
        }
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

    dim3 threadsPerBlock(TILE_SIZE, BLOCK_ROWS);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid optimized triangular matrix multiplication (CUDA)");
}