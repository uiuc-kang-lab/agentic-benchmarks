#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;

    // For lower triangular matrices, elements where row < col are zero
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;

    // Determine tile range relevant for computation
    int t_start = col / TILE_SIZE;
    int t_end   = row / TILE_SIZE;

    #pragma unroll
    for (int t = t_start; t <= t_end; t++) {
        // Load A tile: only load if the column index is within bounds and valid for lower triangular
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (a_col < N && a_col <= row)
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile: only load if the row index is within bounds and meets triangular condition
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && b_row >= col)
            shared_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Determine the effective k range in this tile
        int k_begin = t * TILE_SIZE;
        if (k_begin < col) k_begin = col;
        int k_end = (t + 1) * TILE_SIZE;
        if (k_end > row + 1) k_end = row + 1;

        // If the entire tile is available, unroll fully for maximum performance
        if (k_end - k_begin == TILE_SIZE) {
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++) {
                sum += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
            }
        } else {
            #pragma unroll
            for (int k = k_begin; k < k_end; k++) {
                int local_k = k - t * TILE_SIZE;
                sum += shared_A[threadIdx.y][local_k] * shared_B[local_k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA) with balanced workload");
}
