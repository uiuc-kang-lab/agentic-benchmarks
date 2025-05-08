#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Define tile size for both A and B tiles
#define TILE_SIZE 16

// CUDA kernel employing double buffering in shared memory
__global__ void matmul_db_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int K, int N) {
    // Compute global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Number of tiles required along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate two buffers in shared memory for double buffering
    __shared__ float sA[2][TILE_SIZE][TILE_SIZE];
    __shared__ float sB[2][TILE_SIZE][TILE_SIZE];

    // Buffers indices: current tile index and next tile index
    int curr_buffer = 0;
    int next_buffer = 1;

    // Pre-load the first tile (t = 0) into the current buffer
    int t = 0;
    int a_col = t * TILE_SIZE + threadIdx.x;
    int b_row = t * TILE_SIZE + threadIdx.y;

    if(row < M && a_col < K)
        sA[curr_buffer][threadIdx.y][threadIdx.x] = A[row * K + a_col];
    else
        sA[curr_buffer][threadIdx.y][threadIdx.x] = 0.0f;

    if(b_row < K && col < N)
        sB[curr_buffer][threadIdx.y][threadIdx.x] = B[b_row * N + col];
    else
        sB[curr_buffer][threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Loop over tiles with double buffering
    for(t = 0; t < numTiles - 1; t++) {
        // Pre-fetch next tile into next_buffer
        int a_next_col = (t + 1) * TILE_SIZE + threadIdx.x;
        int b_next_row = (t + 1) * TILE_SIZE + threadIdx.y;

        if(row < M && a_next_col < K)
            sA[next_buffer][threadIdx.y][threadIdx.x] = A[row * K + a_next_col];
        else
            sA[next_buffer][threadIdx.y][threadIdx.x] = 0.0f;

        if(b_next_row < K && col < N)
            sB[next_buffer][threadIdx.y][threadIdx.x] = B[b_next_row * N + col];
        else
            sB[next_buffer][threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product using the current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[curr_buffer][threadIdx.y][k] * sB[curr_buffer][k][threadIdx.x];
        }

        __syncthreads();

        // Swap buffers: next buffer becomes current for the next iteration
        int temp = curr_buffer;
        curr_buffer = next_buffer;
        next_buffer = temp;
    }

    // Compute the last tile using the data in the current buffer
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += sA[curr_buffer][threadIdx.y][k] * sB[curr_buffer][k][threadIdx.x];
    }

    // Write the result to global memory
    if(row < M && col < N)
        C[row * N + col] = sum;
}

// Host function interfacing with PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Get matrix dimensions
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor
    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matmul_db_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with double buffering (CUDA)");
}
