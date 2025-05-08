#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void optimized_tiled_triangular_mm_kernel(
    const float* __restrict__ matrix_a,
    const float* __restrict__ matrix_b,
    float* __restrict__ output,
    int matrix_size
) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = static_cast<int>(blockIdx.y) * TILE_SIZE + static_cast<int>(threadIdx.y);
    int col = static_cast<int>(blockIdx.x) * TILE_SIZE + static_cast<int>(threadIdx.x);

    float sum = 0.0f;

    for (int t = 0; t <= (matrix_size - 1) / TILE_SIZE; ++t) {
        int tiled_row = t * TILE_SIZE + static_cast<int>(threadIdx.y);
        int tiled_col = t * TILE_SIZE + static_cast<int>(threadIdx.x);

        // Load tile from matrix_a with triangular constraint
        if (row < matrix_size && tiled_col < matrix_size && row >= tiled_col) {
            tile_a[threadIdx.y][threadIdx.x] = __ldg(&matrix_a[row * matrix_size + tiled_col]);
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from matrix_b with triangular constraint and transpose storage
        if (tiled_row < matrix_size && col < matrix_size && tiled_row >= col) {
            tile_b[threadIdx.x][threadIdx.y] = __ldg(&matrix_b[tiled_row * matrix_size + col]);
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[threadIdx.x][k];
        }
        __syncthreads();
    }

    if (row < matrix_size && col < matrix_size) {
        output[row * matrix_size + col] = (row >= col) ? sum : 0.0f;
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    optimized_tiled_triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized tiled triangular matrix multiplication (CUDA)");
}