#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ __launch_bounds__(1024, 1) void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             const int N) {
    // Declare shared memory with padding to avoid bank conflicts
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    // Simplified boundary checks
    const bool row_valid = row < N;
    const bool col_valid = col < N;

    // Use registers for accumulation
    float value = 0.0f;
    float a_reg, b_reg;

    // Process tiles
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        const int tile_offset = t * TILE_SIZE;
        
        // Warp-aligned collaborative loading
        if (row_valid && col_valid) {
            s_A[ty][tx] = (row < N && tile_offset + tx < N) ? 
                         A[row * N + tile_offset + tx] : 0.0f;
            s_B[ty][tx] = (tile_offset + ty < N && col < N) ? 
                         B[(tile_offset + ty) * N + col] : 0.0f;
        }

        __syncthreads();

        if (row_valid && col_valid) {
            // Vectorized computation using 4-element chunks
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k += 4) {
                value = __fmaf_rn(s_A[ty][k], s_B[k][tx], value);
                value = __fmaf_rn(s_A[ty][k+1], s_B[k+1][tx], value);
                value = __fmaf_rn(s_A[ty][k+2], s_B[k+2][tx], value);
                value = __fmaf_rn(s_A[ty][k+3], s_B[k+3][tx], value);
            }
        }

        __syncthreads();
    }

    // Warp-uniform write condition
    if (row_valid && col_valid && row < N && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrix dimensions must match");

    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-aligned Matrix Multiplication");
}