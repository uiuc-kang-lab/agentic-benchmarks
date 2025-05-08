#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128  // 4 warps per block
#define TILE_SIZE 16    // Size of the tile each warp processes

__inline__ __device__
float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate warp and lane IDs
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Calculate global position
    const int block_row = blockIdx.y * TILE_SIZE;
    const int block_col = blockIdx.x * TILE_SIZE;
    
    // Each warp processes a 16x16 tile
    const int row = block_row + (warp_id * TILE_SIZE) / TILE_SIZE + lane_id / TILE_SIZE;
    const int col = block_col + lane_id % TILE_SIZE;

    if (row < N && col < N) {
        float sum = 0.0f;
        
        if (row >= col) {
            // Process elements in registers using warp-level collaboration
            #pragma unroll 4
            for (int k = col; k <= row; k++) {
                const float a_val = A[row * N + k];
                const float b_val = B[k * N + col];
                sum += a_val * b_val;
            }
            
            // Use warp shuffle to reduce partial sums if needed
            if (col % TILE_SIZE != 0) {
                sum = warpReduceSum(sum);
            }
            
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

__global__ void triangular_mm_kernel_large(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         const int N) {
    __shared__ float s_partial[BLOCK_SIZE];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.y * TILE_SIZE + warp_id;
    const int col = blockIdx.x * TILE_SIZE + lane_id;

    if (row < N && col < N) {
        float sum = 0.0f;
        
        if (row >= col) {
            // Process in chunks of WARP_SIZE
            for (int k = col; k <= row; k += WARP_SIZE) {
                float a_val = 0.0f, b_val = 0.0f;
                
                if (k + lane_id <= row) {
                    a_val = A[row * N + (k + lane_id)];
                    b_val = B[(k + lane_id) * N + col];
                }
                
                #pragma unroll
                for (int offset = 0; offset < WARP_SIZE; ++offset) {
                    float a_broadcast = __shfl_sync(0xffffffff, a_val, offset);
                    float b_broadcast = __shfl_sync(0xffffffff, b_val, offset);
                    if (k + offset <= row)
                        sum += a_broadcast * b_broadcast;
                }
            }
            
            // Warp-level reduction
            sum = warpReduceSum(sum);
            
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

    const int num_blocks = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(num_blocks, num_blocks);
    
    if (N <= 512) {
        // Use basic kernel for smaller matrices
        triangular_mm_kernel<<<grid, BLOCK_SIZE>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N
        );
    } else {
        // Use optimized kernel for larger matrices
        triangular_mm_kernel_large<<<grid, BLOCK_SIZE>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N
        );
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}