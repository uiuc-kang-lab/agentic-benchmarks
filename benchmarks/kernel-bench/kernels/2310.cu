#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions and warp size
#define TILE_M 8
#define TILE_N 4
#define TILE_K 32
#define WARP_SIZE 32

// This hybrid kernel uses shared memory tiling (from kernel 2) to optimize global memory loads
// and warp-level reduction (from kernel 1) to compute dot products with minimal synchronization.
// Each block computes an output tile of size TILE_M x TILE_N. Within the block, each warp (32 threads)
// computes one element of the tile. The warps load a shared tile of A and B from global memory
// covering a chunk of the K dimension and then each warp cooperatively processes this tile using
// per-lane accumulation and a __shfl_down_sync based reduction. Only lane 0 in each warp writes the
// accumulated result to the output matrix C.

__global__ void matmul_hybrid_shared_warp_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int M, int N, int K) {
    // Compute a linear thread index
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;   // Each warp has 32 threads
    int lane = tid % WARP_SIZE;

    // Map each warp in the block to one output element within the tile
    // The block tile dimensions are TILE_M rows and TILE_N columns
    int warp_row = warp_id / TILE_N;  // row index in block tile [0, TILE_M-1]
    int warp_col = warp_id % TILE_N;  // column index in block tile [0, TILE_N-1]

    // Global indices for the computed output element
    int m = blockIdx.y * TILE_M + warp_row;
    int n = blockIdx.x * TILE_N + warp_col;

    // Register to accumulate the dot product result
    float sum = 0.0f;

    // Declare shared memory tiles for A and B
    // For A: load a tile of shape [TILE_M][TILE_K]
    // For B: load a tile of shape [TILE_N][TILE_K] (B is stored in row-major order with shape (N, K))
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    // Compute the number of tiles needed to cover the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;

    // Loop over the tiles in the K dimension
    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A into shared memory
        // A tile covers rows: blockIdx.y * TILE_M to blockIdx.y * TILE_M + TILE_M - 1
        // and columns: t * TILE_K to t * TILE_K + TILE_K - 1
        int a_tile_size = TILE_M * TILE_K;  // total elements in A tile
        for (int idx = tid; idx < a_tile_size; idx += blockDim.x * blockDim.y) {
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            int global_row = blockIdx.y * TILE_M + row;
            int global_col = t * TILE_K + col;
            if (global_row < M && global_col < K)
                As[row][col] = A[global_row * K + global_col];
            else
                As[row][col] = 0.0f;
        }
        
        // Load a tile of B into shared memory
        // B tile covers rows: blockIdx.x * TILE_N to blockIdx.x * TILE_N + TILE_N - 1
        // and columns: t * TILE_K to t * TILE_K + TILE_K - 1
        int b_tile_size = TILE_N * TILE_K;  
        for (int idx = tid; idx < b_tile_size; idx += blockDim.x * blockDim.y) {
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            int global_row = blockIdx.x * TILE_N + row;  // B's row index corresponds to output column n
            int global_col = t * TILE_K + col;
            if (global_row < N && global_col < K)
                Bs[row][col] = B[global_row * K + global_col];
            else
                Bs[row][col] = 0.0f;
        }

        // Synchronize to ensure shared memory loads are complete
        __syncthreads();

        // Only proceed if the output indices are within the valid range
        if (m < M && n < N) {
            float partial = 0.0f;
            // Each warp computes a portion of the dot product, with each thread processing a subset
            // of the TILE_K elements. The loop strides by WARP_SIZE to distribute the work among lanes.
            for (int k = lane; k < TILE_K; k += WARP_SIZE) {
                partial += As[warp_row][k] * Bs[warp_col][k];
            }
            // Perform warp-level reduction to sum up partial results across the warp
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            // Only the first lane in the warp accumulates the reduced value into the final sum
            if (lane == 0) {
                sum += partial;
            }
        }
        
        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result from lane 0 of each warp to global memory
    if (m < M && n < N && lane == 0) {
        C[m * N + n] = sum;
    }
}


// Forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Launch configuration: each block computes a TILE_M x TILE_N output tile.
    // Block dimensions are set to 32x32 threads (1024 threads per block), which gives exactly TILE_M*TILE_N=32 warps.
    dim3 block(32, 32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_hybrid_shared_warp_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid matrix multiplication with shared memory and warp reduction (CUDA)");
}
