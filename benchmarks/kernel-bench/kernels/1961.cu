#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_ROWS (TILE_SIZE/2)  // 16 rows per block
#define BLOCK_COLS TILE_SIZE      // 32 columns per block

__global__ void hybrid_triangular_mm_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    // Each block handles a BLOCK_ROWS x BLOCK_COLS tile of output
    // Using shared memory for tiles but with warp-level reduction for computation
    
    __shared__ float sA[BLOCK_ROWS][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][BLOCK_COLS];
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    int block_row = blockIdx.y * BLOCK_ROWS;
    int block_col = blockIdx.x * BLOCK_COLS;

    // Early exit for full upper-triangular blocks
    if (block_row + BLOCK_ROWS <= block_col) {
        int row = block_row + (threadIdx.x / BLOCK_COLS);
        int col = block_col + (threadIdx.x % BLOCK_COLS);
        if (row < N && col < N) {
            C[row * N + col] = 0;
        }
        return;
    }

    // Each warp handles 8 output elements (arranged in a 2x4 pattern)
    int warp_row = warp_id / 2;              // 0 or 1
    int warp_col = (warp_id % 2) * 16;       // 0 or 16
    int row = block_row + warp_row * 8 + (lane / 4);
    int col = block_col + warp_col + (lane % 4);

    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0;
        return;
    }

    float sum = 0;
    
    // Process input matrix in tiles
    for (int m = 0; m < N; m += TILE_SIZE) {
        // Collaborative loading of tiles into shared memory
        for (int i = threadIdx.x; i < BLOCK_ROWS * TILE_SIZE; i += blockDim.x) {
            int r = block_row + i / TILE_SIZE;
            int k = m + i % TILE_SIZE;
            sA[i / TILE_SIZE][i % TILE_SIZE] = (r < N && k < N && r >= k) ? A[r * N + k] : 0;
        }
        
        for (int i = threadIdx.x; i < TILE_SIZE * BLOCK_COLS; i += blockDim.x) {
            int k = m + i / BLOCK_COLS;
            int c = block_col + i % BLOCK_COLS;
            sB[i / BLOCK_COLS][i % BLOCK_COLS] = (k < N && c < N && k >= c) ? B[k * N + c] : 0;
        }
        
        __syncthreads();

        // Compute valid k range for this tile
        int k_start = max(col, m);
        int k_end = min(min(row + 1, m + TILE_SIZE), N);
        
        // Each thread processes its assigned k values
        for (int k = k_start; k < k_end; k++) {
            sum += sA[row - block_row][k - m] * sB[k - m][col - block_col];
        }

        __syncthreads();
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write result
    if (lane == 0) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 grid((N + BLOCK_COLS - 1)/BLOCK_COLS, (N + BLOCK_ROWS - 1)/BLOCK_ROWS);
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);  // 128 threads per block
    
    hybrid_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}