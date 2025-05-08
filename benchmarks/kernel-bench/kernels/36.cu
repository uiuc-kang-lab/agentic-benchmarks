#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")


// Double-buffered tiled matrix multiplication kernel
__global__ void matmul_double_buffered_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    // Two buffers for double buffering in shared memory
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    if (row >= N || col >= N) return;

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    float C_value = 0.0f;
    int curBuf = 0, nextBuf = 1;

    // Preload the first tile into the current buffer
    {
        int m = 0;
        int A_col = m * TILE_SIZE + tx;
        int B_row = m * TILE_SIZE + ty;
        As[curBuf][ty][tx] = (row < N && A_col < N) ? A[row * N + A_col] : 0.0f;
        Bs[curBuf][ty][tx] = (col < N && B_row < N) ? B[B_row * N + col] : 0.0f;
    }
    __syncthreads();

    // Loop over tiles using double buffering
    for (int m = 0; m < num_tiles - 1; m++) {
        // Prefetch the next tile into the alternate buffer
        int A_col = (m + 1) * TILE_SIZE + tx;
        int B_row = (m + 1) * TILE_SIZE + ty;
        As[nextBuf][ty][tx] = (row < N && A_col < N) ? A[row * N + A_col] : 0.0f;
        Bs[nextBuf][ty][tx] = (col < N && B_row < N) ? B[B_row * N + col] : 0.0f;
        
        __syncthreads();

        // Compute partial product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            C_value += As[curBuf][ty][k] * Bs[curBuf][k][tx];
        }

        // Swap buffers for next iteration
        curBuf = 1 - curBuf;
        nextBuf = 1 - nextBuf;

        __syncthreads();
    }

    // Process the last preloaded tile
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        C_value += As[curBuf][ty][k] * Bs[curBuf][k][tx];
    }

    C[row * N + col] = C_value;
}


torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);

    auto C = torch::zeros({N, N}, A.options());
    
    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_double_buffered_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Double-buffered Matrix multiplication kernel (CUDA)");
}
