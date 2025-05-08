#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void optimized_coalesced_triangular_mm_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
    float* __restrict__ matrix_c,
    const int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int row = static_cast<int>(blockIdx.y) * TILE_SIZE + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * TILE_SIZE + static_cast<int>(threadIdx.x);
    const bool valid_thread = (row < N && col < N && row >= col);

    float sum = 0.0f;
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        const int tile_row = row;
        const int tile_col = t * TILE_SIZE + static_cast<int>(threadIdx.x);
        const int tile_idx = t * TILE_SIZE + static_cast<int>(threadIdx.y);

        // Load tile from A - only load if within lower triangular region
        As[threadIdx.y][threadIdx.x] = (tile_row < N && tile_col < N && tile_row >= tile_col) 
            ? __ldg(&matrix_a[tile_row * N + tile_col]) 
            : 0.0f;

        // Load tile from B - only load if within lower triangular region
        Bs[threadIdx.y][threadIdx.x] = (tile_idx < N && col < N && tile_idx >= col) 
            ? __ldg(&matrix_b[tile_idx * N + col]) 
            : 0.0f;

        // Single synchronization point per tile
        __syncthreads();

        // Compute dot product for this tile
        if (valid_thread) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (valid_thread) {
        matrix_c[row * N + col] = sum;
    } else if (row < N && col < N) {
        matrix_c[row * N + col] = 0.0f;
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    const dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    const dim3 block(TILE_SIZE, TILE_SIZE);

    optimized_coalesced_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized coalesced triangular matrix multiplication (CUDA)");
}