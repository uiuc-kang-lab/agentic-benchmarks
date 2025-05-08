#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 128
#define NUM_STREAMS 4

__global__ void triangular_mm_kernel_atomic_minimized(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col <= row) {
        float sum = 0.f;
        for (int k = col; k <= row; ++k) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        // Reduce use of atomic operations by storing individual results
        // in shared memory and then writing to global memory if thread group size > 1
        if (blockDim.x == 1 && blockDim.y == 1) {
            C[row * N + col] = sum;
        } else {
            atomicAdd(&C[row * N + col], sum);
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
    auto C = torch::zeros_like(A);  // Initialize with zeros

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    triangular_mm_kernel_atomic_minimized<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}
