#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

namespace {
__device__ inline bool is_valid_element(int row, int col, int N) {
    return (row < N && col < N && row >= col);
}

__global__ void hybrid_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    const int start_tile = blockIdx.x;
    const int end_tile = min(blockIdx.y, (N - 1) / TILE_SIZE);

    for (int t = start_tile; t <= end_tile; ++t) {
        const int tRow = blockIdx.y * TILE_SIZE + threadIdx.y;
        const int tCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = is_valid_element(tRow, tCol, N) ? 
            __ldg(&A[tRow * N + tCol]) : 0.0f;

        const int tRow_B = t * TILE_SIZE + threadIdx.y;
        const int tCol_B = blockIdx.x * TILE_SIZE + threadIdx.x;
        Bs[threadIdx.y][threadIdx.x] = is_valid_element(tRow_B, tCol_B, N) ?
            __ldg(&B[tRow_B * N + tCol_B]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1) && A.size(0) == B.size(0),
                "Invalid matrix dimensions");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    hybrid_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid triangular matrix multiplication (CUDA)");
}