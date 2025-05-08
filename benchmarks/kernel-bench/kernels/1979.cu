#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define WARP_SIZE 32

__global__ void triangular_mm_kernel_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int lane_id = threadIdx.y * TILE_SIZE + threadIdx.x;
    const int warp_id = lane_id / WARP_SIZE;
    const int lane = lane_id % WARP_SIZE;
    
    float sum = 0.0f;
    
    // Process tiles along the k dimension
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaborative loading of A and B tiles into shared memory
        const int tile_idx = t * TILE_SIZE;
        if (row < N && (tile_idx + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tile_idx + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((tile_idx + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile_idx + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        // Compute partial sums for this tile
        if (row < N && col < N) {
            if (row >= col) {  // Lower triangular part only
                int k_start = max(tile_idx, col);
                int k_end = min(tile_idx + TILE_SIZE, row + 1);
                if (k_start < k_end) {
                    // Map global k indices to tile-local indices
                    k_start -= tile_idx;
                    k_end -= tile_idx;
                    
                    // Compute partial sum for this tile
                    #pragma unroll
                    for (int k = k_start; k < k_end; ++k) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Warp-level reduction for better efficiency
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Write result
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        } else {
            // Only the first lane in each warp writes the final result
            if (lane == 0) {
                C[row * N + col] = sum;
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Calculate shared memory size
    const int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    
    triangular_mm_kernel_shared<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared memory optimized triangular matrix multiplication (CUDA)");
}