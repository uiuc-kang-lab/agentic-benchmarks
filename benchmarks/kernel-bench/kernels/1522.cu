#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Stride loop for output matrix coverage
    for (int row = blockIdx.y * blockDim.y + ty; row < N; row += blockDim.y * gridDim.y) {
        for (int col = blockIdx.x * blockDim.x + tx; col < N; col += blockDim.x * gridDim.x) {
            float value = 0;

            for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {
                // Load tile from A with boundary check
                int a_col = i * TILE_SIZE + tx;
                s_A[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

                // Load tile from B with boundary check
                int b_row = i * TILE_SIZE + ty;
                s_B[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

                __syncthreads();

                // Accumulate product
                for (int k = 0; k < TILE_SIZE; ++k)
                    value += s_A[ty][k] * s_B[k][tx];

                __syncthreads();
            }

            // Write result with boundary check
            if (row < N && col < N)
                C[row * N + col] = value;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    // Reduce grid dimensions to force stride loops
    grid.x = min(grid.x, 65535);
    grid.y = min(grid.y, 65535);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride-optimized Matrix Multiplication");
}