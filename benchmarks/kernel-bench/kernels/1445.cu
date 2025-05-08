#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_MATRIX_DIM 8192

// Use constant memory to hold frequently used parameters
__constant__ int d_N;
__constant__ int d_num_tiles;

// This kernel uses double buffering in shared memory to overlap the loading of the next tile with the computation of the current tile
// Two shared memory buffers are used for A and B alternatively.

__global__ void matmul_kernel_double_buffer(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C) {
    // Declare double buffered shared memory
    __shared__ float s_A[2][TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[2][TILE_SIZE][TILE_SIZE];

    // Compute thread-local indices
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float value = 0.0f;
    
    // Current buffer index
    int cur = 0;

    // Preload the first tile (m = 0) into buffer 0
    int tile_col = 0 * TILE_SIZE + tx;
    int tile_row = 0 * TILE_SIZE + ty;
    if(row < d_N && tile_col < d_N)
        s_A[cur][ty][tx] = A[row * d_N + tile_col];
    else
        s_A[cur][ty][tx] = 0.0f;

    if(tile_row < d_N && col < d_N)
        s_B[cur][ty][tx] = B[tile_row * d_N + col];
    else
        s_B[cur][ty][tx] = 0.0f;

    __syncthreads();

    // Loop over all tiles
    for (int m = 0; m < d_num_tiles; m++) {
        int nxt = 1 - cur;  // alternate buffer index
        // Prefetch the next tile if it exists
        if(m < d_num_tiles - 1) {
            int next_tile = m + 1;
            int next_col = next_tile * TILE_SIZE + tx;
            int next_row = next_tile * TILE_SIZE + ty;
            
            if(row < d_N && next_col < d_N)
                s_A[nxt][ty][tx] = A[row * d_N + next_col];
            else
                s_A[nxt][ty][tx] = 0.0f;

            if(next_row < d_N && col < d_N)
                s_B[nxt][ty][tx] = B[next_row * d_N + col];
            else
                s_B[nxt][ty][tx] = 0.0f;
        }

        // Ensure that the current tile has been loaded
        __syncthreads();

        // Compute partial results using the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += s_A[cur][ty][k] * s_B[cur][k][tx];
        }

        // Swap buffers if there is a next tile
        if(m < d_num_tiles - 1) {
            cur = nxt;
        }

        // Make sure that prefetching is complete before next iteration
        __syncthreads();
    }

    // Write the result
    if(row < d_N && col < d_N) {
        C[row * d_N + col] = value;
    }
}


// The forward function launches the kernel. It uses constant memory to pass over matrix size parameters
// and adopts a tiled configuration with a fixed block size of TILE_SIZE*TILE_SIZE threads.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    const int N = A.size(0);
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Copy constants to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Use a 1D configuration of threads: each block has TILE_SIZE*TILE_SIZE threads
    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    dim3 threads(threads_per_block);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_double_buffer<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Double Buffering (CUDA)");
}
