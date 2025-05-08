#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void coalesced_triangular_mm_kernel(
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
        if (tRow < N && tCol < N && tRow >= tCol)
            As[threadIdx.y][threadIdx.x] = __ldg(&A[tRow * N + tCol]);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tRow = t * TILE_SIZE + static_cast<int>(threadIdx.y);
        tCol = static_cast<int>(blockIdx.x) * TILE_SIZE + static_cast<int>(threadIdx.x);
        if (tRow < N && tCol < N && tRow >= tCol)
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[tRow * N + tCol]);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        if (row >= col) C[row * N + col] = sum;
        else C[row * N + col] = 0.0f;
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

    coalesced_triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced triangular matrix multiplication (CUDA)");
}