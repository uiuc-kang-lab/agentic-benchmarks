#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Kernel that uses a grid-stride loop to evenly distribute the workload
// across threads and blocks. Each thread computes one or more elements of C.
__global__ void GridStrideMatmulKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int K, int N) {
    int totalElements = M * N;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements in a grid-stride loop
    for (int idx = tid; idx < totalElements; idx += stride) {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Forward function wraps the kernel launch and provides fallback for error checking
// and input verification. This kernel is specifically designed for the
// small K dimension scenario, ensuring all threads participate evenly in the
// computation of the output matrix.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Choose a reasonable number of threads and blocks to evenly cover M*N elements
    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    GridStrideMatmulKernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Grid-stride matrix multiplication with evenly distributed workload (CUDA)");
}
