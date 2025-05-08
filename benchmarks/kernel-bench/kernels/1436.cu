#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// This kernel partitions the inner dimension (k) across a 3D grid (using blockIdx.z).
// Each block computes a partial dot product for its tile and then atomically accumulates the result
// into the global output matrix. Atomic operations are used only once per thread (after full
// accumulation over the tile segment), thereby minimizing global memory contention.

__global__ void matmul_kernel_atomic(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int N) {
    // Use blockIdx.x and blockIdx.y to index the output tile, and blockIdx.z to index the k-split
    int tile_row = blockIdx.y;  // output tile row index
    int tile_col = blockIdx.x;  // output tile col index
    int tile_k   = blockIdx.z;  // k-dimension tile index

    // Compute the global row and column for this thread
    int row = tile_row * TILE_SIZE + threadIdx.y;
    int col = tile_col * TILE_SIZE + threadIdx.x;
    
    // Compute starting index for the k-dimension for this block
    int k_start = tile_k * TILE_SIZE;

    float sum = 0.0f;

    // Shared memory to load tiles of A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Each thread loads one element from A and one from B corresponding to the current k-tile.
    int a_col = k_start + threadIdx.x;  // column index for A
    if (row < N && a_col < N) {
        s_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
    } else {
        s_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    int b_row = k_start + threadIdx.y;  // row index for B
    if (b_row < N && col < N) {
        s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
    } else {
        s_B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Determine the actual dimension of the k-tile (it may be smaller near boundaries)
    int k_dim = TILE_SIZE;
    if (k_start + TILE_SIZE > N) {
        k_dim = N - k_start;
    }

    // Perform the multiplication over the k-dimension for this tile
    #pragma unroll
    for (int i = 0; i < k_dim; i++) {
        sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
    }

    // Combine partial results from different k-tiles using atomic addition
    if (row < N && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    // Determine how many k-tiles are needed to cover the inner dimension
    int k_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    // Initialize the output tensor C to zeros. It will be accumulated via atomicAdd.
    auto C = torch::zeros({N, N}, options);

    // Set up grid dimensions: 
    // - blocks.x and blocks.y cover the output tile decomposition
    // - blocks.z splits the k-dimension so that each block computes a partial sum
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE,
                k_tiles);

    matmul_kernel_atomic<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Atomic Reduction (CUDA)");
}
