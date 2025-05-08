#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void triangular_mm_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col <= row) {
        for (int tile = 0; tile < (row + TILE_SIZE) / TILE_SIZE; ++tile) {
            int tile_offset = tile * TILE_SIZE;

            // Load A tile (row-wise)
            if (threadIdx.x < TILE_SIZE) {
                int a_col = tile_offset + threadIdx.x;
                As[threadIdx.y][threadIdx.x] = (a_col <= row) ? A[row * N + a_col] : 0.0f;
            }

            // Load B tile (column-wise with coalescing)
            if (threadIdx.y < TILE_SIZE) {
                int b_row = tile_offset + threadIdx.y;
                Bs[threadIdx.y][threadIdx.x] = (b_row >= col && b_row <= row) ? B[b_row * N + col] : 0.0f;
            }

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            __syncthreads();
        }

        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
    }
}

torch::Tensor forward_shared(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_shared_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward_shared, "Tiled shared memory triangular matmul (CUDA)");
}
