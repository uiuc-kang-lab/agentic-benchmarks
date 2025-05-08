#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float shared_A[WARP_SIZE][WARP_SIZE];
    __shared__ float shared_B[WARP_SIZE][WARP_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.f;

    for (int tile = 0; tile < (N + WARP_SIZE - 1) / WARP_SIZE; ++tile) {
        int tiled_k = tile * WARP_SIZE + threadIdx.x;

        if (row < N && tiled_k < N)
            shared_A[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + tiled_k]);
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.f;

        if (col < N && tiled_k < N)
            shared_B[threadIdx.y][threadIdx.x] = __ldg(&B[tiled_k * N + col]);
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        for (int k = 0; k < WARP_SIZE; ++k) {
            if (tiled_k + k < N && row >= col) {
                sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row >= col) {
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.f;
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

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(WARP_SIZE, WARP_SIZE);
    dim3 numBlocks((N + WARP_SIZE - 1) / WARP_SIZE, (N + WARP_SIZE - 1) / WARP_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular Matrix Multiplication with shared memory tiling (CUDA)");
}