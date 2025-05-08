#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable parameters
const int TILE_SIZE = 32;
const int WARPS_PER_BLOCK = 4;
const int BLOCK_ROWS = 4;
const int BLOCK_COLS = 4;

__global__ void hybrid_matmul_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int M, const int N, const int K) {
    // Shared memory for block-level tiling
    __shared__ float As[BLOCK_ROWS][TILE_SIZE];
    __shared__ float Bs[BLOCK_COLS][TILE_SIZE];
    
    // Thread and block indexing
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int warp_row = warp_id / BLOCK_COLS;
    const int warp_col = warp_id % BLOCK_COLS;
    
    // Global matrix positions
    const int row = blockIdx.y * BLOCK_ROWS + warp_row;
    const int col = blockIdx.x * BLOCK_COLS + warp_col;

    // Accumulator for dot product
    float sum = 0.0f;

    // Process K dimension in tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        const int k_offset = t * TILE_SIZE;
        
        // Collaborative loading of A and B tiles into shared memory
        #pragma unroll
        for (int i = lane_id; i < BLOCK_ROWS * TILE_SIZE; i += TILE_SIZE) {
            const int shared_row = i / TILE_SIZE;
            const int shared_col = i % TILE_SIZE;
            if (row < M && (k_offset + shared_col) < K) {
                As[shared_row][shared_col] = A[row * K + k_offset + shared_col];
            } else {
                As[shared_row][shared_col] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int i = lane_id; i < BLOCK_COLS * TILE_SIZE; i += TILE_SIZE) {
            const int shared_row = i / TILE_SIZE;
            const int shared_col = i % TILE_SIZE;
            if (col < N && (k_offset + shared_col) < K) {
                Bs[shared_row][shared_col] = B[col * K + k_offset + shared_col];
            } else {
                Bs[shared_row][shared_col] = 0.0f;
            }
        }
        
        __syncthreads();

        // Compute partial dot products using warp-level parallelism
        float temp = 0.0f;
        #pragma unroll
        for (int k = lane_id; k < TILE_SIZE; k += 32) {
            temp += As[warp_row][k] * Bs[warp_col][k];
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            temp += __shfl_down_sync(0xffffffff, temp, offset);
        }

        if (lane_id == 0) {
            sum += temp;
        }
        
        __syncthreads();
    }

    // Write result
    if (lane_id == 0 && row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int N = B.size(0);
    const int K = A.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 grid((N + BLOCK_COLS - 1) / BLOCK_COLS, 
              (M + BLOCK_ROWS - 1) / BLOCK_ROWS);
    dim3 block(32, BLOCK_ROWS * BLOCK_COLS);

    hybrid_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid matrix multiplication with transposed B");
}