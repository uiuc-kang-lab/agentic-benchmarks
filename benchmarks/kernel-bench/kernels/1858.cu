#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int d_N;  // Matrix size in constant memory

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate row and column from linear index using constant memory N
    const int row = tid / d_N;
    const int col = tid % d_N;
    
    if (row < d_N && col < d_N) {
        if (row < col) {
            C[row * d_N + col] = 0.f;
        } else {
            float sum = 0.f;
            #pragma unroll 16
            for (int k = col; k <= row; ++k) {
                sum += A[row * d_N + k] * B[k * d_N + col];
            }
            C[row * d_N + col] = sum;
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Copy N to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    // Use 1D thread configuration with optimal block size
    const int threadsPerBlock = 256;
    const int numElements = N * N;
    const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}