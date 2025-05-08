#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<int BLOCK_SIZE>
__global__ void triangular_mm_kernel_dynamic(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           const int N) {
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            #pragma unroll 4
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
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

    // Choose block size based on matrix size
    // For larger matrices, use larger block sizes
    const int BLOCK_SIZE = (N >= 1024) ? 64 : 
                          (N >= 512) ? 32 : 16;

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    switch(BLOCK_SIZE) {
        case 64:
            triangular_mm_kernel_dynamic<64><<<numBlocks, threadsPerBlock>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
            break;
        case 32:
            triangular_mm_kernel_dynamic<32><<<numBlocks, threadsPerBlock>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
            break;
        default:
            triangular_mm_kernel_dynamic<16><<<numBlocks, threadsPerBlock>>>(
                A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Dynamic block size triangular matrix multiplication (CUDA)");
}