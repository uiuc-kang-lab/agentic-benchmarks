#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    int block_row_start = blockIdx.y * blockDim.y;
    int block_col_start = blockIdx.x * blockDim.x;
    int block_row_end = block_row_start + blockDim.y - 1;
    int block_col_end = block_col_start + blockDim.x - 1;

    if (block_row_end < block_col_start) {
        C[row * N + col] = 0.f;
        return;
    }

    if (block_row_start >= block_col_end) {
        float sum = 0.f;
        #pragma unroll 4
        for (int k = col; k <= row; ++k) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        C[row * N + col] = sum;
        return;
    }

    int warp_row = row & ~(WARP_SIZE - 1);
    int warp_col = col & ~(WARP_SIZE - 1);
    if (warp_row < warp_col) {
        C[row * N + col] = 0.f;
        return;
    }

    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;
    #pragma unroll 4
    for (int k = col; k <= row; ++k) {
        sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
    }
    C[row * N + col] = sum;
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
    m.def("forward", &forward, "Optimized Triangular Matrix Multiplication with alignment and __ldg() (CUDA)");
}