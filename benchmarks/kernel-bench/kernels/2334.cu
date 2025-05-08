#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4

__global__ void matmul_balanced_workload_kernel(const float* A, const float* B, float* C, 
                                              const int M, const int N, const int K) {
    // Calculate base indices for this thread
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int by = blockIdx.y * BLOCK_SIZE * ELEMENTS_PER_THREAD;

    // Compute multiple output elements per thread
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            const int row = by + ty * ELEMENTS_PER_THREAD + i;
            const int col = bx + tx * ELEMENTS_PER_THREAD + j;

            if (row < M && col < N) {
                float sum = 0.0f;
                
                // Process the K dimension in chunks to improve cache utilization
                for (int k = 0; k < K; ++k) {
                    sum += __ldg(&A[row * K + k]) * __ldg(&B[col * K + k]);
                }
                
                C[row * N + col] = sum;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Calculate grid dimensions to account for multiple elements per thread
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD),
              (M + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD));

    matmul_balanced_workload_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using balanced workload (CUDA)");
}