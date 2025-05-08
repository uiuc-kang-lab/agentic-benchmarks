#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

namespace {
__device__ inline bool thread_in_bounds(int row, int col, int N) {
    return (row < N && col < N);
}

__device__ inline float compute_element(
    const float* __restrict__ A,
    const float* __restrict__ B,
    int row,
    int col,
    int N
) {
    float sum = 0.0f;
    #pragma unroll
    for(int k = col; k <= row; ++k) {
        sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
    }
    return sum;
}
} // anonymous namespace

__global__ void uniform_control_flow_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = static_cast<int>(blockIdx.y) * TILE_SIZE + static_cast<int>(threadIdx.y);
    int col = static_cast<int>(blockIdx.x) * TILE_SIZE + static_cast<int>(threadIdx.x);

    float sum = 0.0f;

    for (int t = static_cast<int>(blockIdx.x); t <= static_cast<int>(blockIdx.y); ++t) {
        int tRow = static_cast<int>(blockIdx.y) * TILE_SIZE + static_cast<int>(threadIdx.y);
        int tCol = t * TILE_SIZE + static_cast<int>(threadIdx.x);
        As[threadIdx.y][threadIdx.x] = (thread_in_bounds(tRow, tCol, N) && row >= tCol) ? __ldg(&A[tRow * N + tCol]) : 0.0f;

        tRow = t * TILE_SIZE + static_cast<int>(threadIdx.y);
        tCol = static_cast<int>(blockIdx.x) * TILE_SIZE + static_cast<int>(threadIdx.x);
        Bs[threadIdx.y][threadIdx.x] = (thread_in_bounds(tRow, tCol, N) && tRow >= col) ? __ldg(&B[tRow * N + tCol]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (thread_in_bounds(row, col, N)) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
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

    uniform_control_flow_triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Uniform control flow triangular matrix multiplication (CUDA)");
}