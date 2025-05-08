#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_MATRIX_DIM 8192

// CUDA kernel using double buffering in shared memory
__global__ void matmul_kernel_double_buffer(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N, int numTiles) {
    // Allocate two buffers in shared memory for the current and next tiles
    __shared__ float s_A[2][TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Use ping-pong buffering: currBuf is the index of the buffer holding the currently computed tile
    int currBuf = 0;

    // Preload the first tile (tile 0) into shared memory buffer currBuf
    int t = 0;
    int aCol = t * TILE_SIZE + threadIdx.x;
    int bRow = t * TILE_SIZE + threadIdx.y;

    s_A[currBuf][threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
    s_B[currBuf][threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

    __syncthreads();

    // Loop over all tiles except the last one using double buffering
    for (t = 0; t < numTiles - 1; t++) {
        int nextBuf = 1 - currBuf;
        int aColNext = (t + 1) * TILE_SIZE + threadIdx.x;
        int bRowNext = (t + 1) * TILE_SIZE + threadIdx.y;
        
        // Preload the next tile into the alternate buffer
        s_A[nextBuf][threadIdx.y][threadIdx.x] = (row < N && aColNext < N) ? A[row * N + aColNext] : 0.0f;
        s_B[nextBuf][threadIdx.y][threadIdx.x] = (bRowNext < N && col < N) ? B[bRowNext * N + col] : 0.0f;

        __syncthreads();

        // Compute partial product using the tile in the current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[currBuf][threadIdx.y][k] * s_B[currBuf][k][threadIdx.x];
        }

        __syncthreads();
        // Swap buffers for the next iteration
        currBuf = nextBuf;
    }

    // Process the last tile which is already loaded in the current buffer
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += s_A[currBuf][threadIdx.y][k] * s_B[currBuf][k][threadIdx.x];
    }

    // Write the result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// C++ interface (Pybind11 binding)

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_double_buffer<<<blocks, threads>>>(
         A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, numTiles);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Double Buffering Matrix Multiplication (CUDA)");
}
