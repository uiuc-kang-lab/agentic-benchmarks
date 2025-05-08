#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Device function to compute dot product for a specific row segment
template<typename T>
__device__ __forceinline__ T compute_dot_product(const T* __restrict__ A,
                                                const T* __restrict__ B,
                                                const int row,
                                                const int col,
                                                const int start,
                                                const int end,
                                                const int N) {
    T sum = 0;
    #pragma unroll 4
    for (int k = start; k < end; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    return sum;
}

// Device function to handle the lower triangular computation
template<typename T>
__device__ __forceinline__ T compute_triangular_element(const T* __restrict__ A,
                                                       const T* __restrict__ B,
                                                       const int row,
                                                       const int col,
                                                       const int N) {
    if (row < col) {
        return 0;
    }
    return compute_dot_product(A, B, row, col, col, row + 1, N);
}

__global__ void triangular_mm_kernel_modular(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           const int N) {
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < N && col < N) {
        C[row * N + col] = compute_triangular_element(A, B, row, col, N);
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel_modular<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Modular triangular matrix multiplication (CUDA)");
}