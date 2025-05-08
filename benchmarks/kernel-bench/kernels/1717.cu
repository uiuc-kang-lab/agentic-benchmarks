#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Calculate total elements in lower triangle
    const int total_elements = (N * (N + 1)) / 2;
    
    // Each thread processes multiple elements with stride
    for (int idx = tid; idx < total_elements; idx += stride) {
        // Convert linear index to 2D coordinates
        // Using reverse mapping for lower triangular matrix
        int row = (int)((-1 + sqrt(1 + 8.0f * idx)) / 2.0f);
        int col = idx - (row * (row + 1)) / 2;
        
        if (row < N && col <= row) {
            float sum = 0.0f;
            // Compute matrix multiplication for this element
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    
    // Handle upper triangle (set to zero)
    for (int idx = tid; idx < N * N; idx += stride) {
        int row = idx / N;
        int col = idx % N;
        if (row < col && row < N && col < N) {
            C[row * N + col] = 0.0f;
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

    // Optimize thread and block count
    const int threadsPerBlock = 32;
    const int numBlocks = min(65535, (N * N + threadsPerBlock - 1) / threadsPerBlock);

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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}