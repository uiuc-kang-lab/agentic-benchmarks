#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Optimized kernel using double buffering to reduce __syncthreads() calls
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    // Allocate double-buffered shared memory for tiles
    __shared__ float shared_A[2][TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Early exit for threads computing upper triangle
    if (row < N && col < N && row < col) {
        C[row * N + col] = 0.0f;
        return;
    }
    if (row >= N || col >= N) return;

    float sum = 0.0f;

    // Determine the range of tiles to process based on triangular structure
    int start_tile = col / TILE_SIZE;
    int end_tile = row / TILE_SIZE;

    // Load the first tile into buffer 0
    int t0 = start_tile;
    int a_col = t0 * TILE_SIZE + threadIdx.x;
    if (row < N && a_col < N && a_col <= row)
        shared_A[0][threadIdx.y][threadIdx.x] = A[row * N + a_col];
    else
        shared_A[0][threadIdx.y][threadIdx.x] = 0.0f;

    int b_row = t0 * TILE_SIZE + threadIdx.y;
    if (b_row < N && col < N && b_row >= col)
        shared_B[0][threadIdx.y][threadIdx.x] = B[b_row * N + col];
    else
        shared_B[0][threadIdx.y][threadIdx.x] = 0.0f;

    // Ensure the first tile is loaded
    __syncthreads();

    int current = 0;
    // Loop over tiles with double buffering
    for (int t = start_tile; t < end_tile; t++) {
        int next = 1 - current;
        int t_next = t + 1;
        
        // Preload next tile into the 'next' buffer concurrently
        int next_a_col = t_next * TILE_SIZE + threadIdx.x;
        if (row < N && next_a_col < N && next_a_col <= row)
            shared_A[next][threadIdx.y][threadIdx.x] = A[row * N + next_a_col];
        else
            shared_A[next][threadIdx.y][threadIdx.x] = 0.0f;

        int next_b_row = t_next * TILE_SIZE + threadIdx.y;
        if (next_b_row < N && col < N && next_b_row >= col)
            shared_B[next][threadIdx.y][threadIdx.x] = B[next_b_row * N + col];
        else
            shared_B[next][threadIdx.y][threadIdx.x] = 0.0f;
        
        // Compute partial sum for the current tile
        int tile_start_k = (t * TILE_SIZE) < col ? col : (t * TILE_SIZE);
        int tile_end_k = ((t + 1) * TILE_SIZE - 1) > row ? row : ((t + 1) * TILE_SIZE - 1);
        for (int k = tile_start_k; k <= tile_end_k; k++) {
            int local_k = k - t * TILE_SIZE;
            sum += shared_A[current][threadIdx.y][local_k] * shared_B[current][local_k][threadIdx.x];
        }
        
        // Synchronize to ensure the next tile is fully loaded
        __syncthreads();
        current = next;
    }
    
    // Process the final tile (no preloading needed)
    int t = end_tile;
    int tile_start_k = (t * TILE_SIZE) < col ? col : (t * TILE_SIZE);
    int tile_end_k = ((t + 1) * TILE_SIZE - 1) > row ? row : ((t + 1) * TILE_SIZE - 1);
    for (int k = tile_start_k; k <= tile_end_k; k++) {
        int local_k = k - t * TILE_SIZE;
        sum += shared_A[current][threadIdx.y][local_k] * shared_B[current][local_k][threadIdx.x];
    }

    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch.
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}
