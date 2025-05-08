#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.f;

    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int tiledRow = tile * BLOCK_SIZE + threadIdx.y;
        int tiledCol = tile * BLOCK_SIZE + threadIdx.x;

        if (tiledRow < N && row < N) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + tiledCol]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.f;
        }

        if (tiledCol < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[tiledRow * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        if (row >= col) {
            #pragma unroll 4
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        if (row >= col) {
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.f;
        }
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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

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
    m.def("forward", &forward, "Optimized Triangular Matrix Multiplication with Shared Memory (CUDA)");
}