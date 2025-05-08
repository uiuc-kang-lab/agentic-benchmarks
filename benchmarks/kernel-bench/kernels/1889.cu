#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 32
#define CHUNK_SIZE 8

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate global indices
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Register arrays for accumulation and data loading
    float reg_C[CHUNK_SIZE] = {0.0f};
    float reg_A[CHUNK_SIZE];
    float reg_B[CHUNK_SIZE];
    
    // Warp index within the block
    const int warp_id = threadIdx.y / WARP_SIZE;
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    
    // Process matrix in chunks
    for (int bk = 0; bk <= (row / BLOCK_SIZE); bk++) {
        const int block_start = bk * BLOCK_SIZE;
        
        // Load data into registers using vectorized loads
        #pragma unroll
        for (int c = 0; c < CHUNK_SIZE; c++) {
            const int k = block_start + c * (WARP_SIZE / CHUNK_SIZE) + lane_id % (WARP_SIZE / CHUNK_SIZE);
            
            if (row < N && k < N) {
                reg_A[c] = A[row * N + k];
            } else {
                reg_A[c] = 0.0f;
            }
            
            if (k < N && col < N) {
                reg_B[c] = B[k * N + col];
            } else {
                reg_B[c] = 0.0f;
            }
        }
        
        // Compute using warp-level communication
        if (row < N && col < N && row >= col) {
            const int k_start = max(block_start, col);
            const int k_end = min(block_start + BLOCK_SIZE, row + 1);
            
            #pragma unroll
            for (int k = k_start; k < k_end; k += CHUNK_SIZE) {
                #pragma unroll
                for (int c = 0; c < CHUNK_SIZE && (k + c) < k_end; c++) {
                    // Use warp shuffle to broadcast values within the warp
                    float a_val = __shfl_sync(0xffffffff, reg_A[c], (k - block_start + c) % WARP_SIZE);
                    float b_val = __shfl_sync(0xffffffff, reg_B[c], lane_id);
                    reg_C[c] += a_val * b_val;
                }
            }
        }
    }
    
    // Reduction within warp using shuffle operations
    if (row < N && col < N) {
        if (row >= col) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < CHUNK_SIZE; i++) {
                sum += reg_C[i];
            }
            
            // Warp-level reduction using shuffle
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            // First thread in warp writes result
            if (lane_id == 0) {
                C[row * N + col] = sum;
            }
        } else {
            C[row * N + col] = 0.0f;
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}