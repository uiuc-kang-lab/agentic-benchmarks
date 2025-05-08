#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 256

// Kernel using 1D thread mapping: each thread computes one element in the output matrix C.
// This mapping simplifies thread/block indexing and ensures that threads access contiguous memory locations in C,
// leading to improved memory coalescing. With a small K dimension, the inner loop is short and can be unrolled effectively.
__global__ void smallKMatmulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// The forward function checks input tensors and launches the kernel.
// It uses a 1D grid mapping over the M x N output elements to optimize thread usage.
// This approach minimizes indexing overhead and ensures that adjacent threads compute adjacent elements, 
// enabling efficient memory transactions.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    int total_elements = M * N;
    int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    smallKMatmulKernel<<<blocks, THREADS_PER_BLOCK>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Small K matrix multiplication with 1D thread mapping (CUDA)");
}
