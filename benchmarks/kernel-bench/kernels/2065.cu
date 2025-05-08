#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Utilize TILE_SIZE constant and a different indexing method for zeroing out upper triangular elements.
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

__global__ void warp_uniform_combined_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int N) {
    // Calculate element index in lower triangular matrix
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * (N + 1) / 2;
    
    if (idx >= total_elements) return;

    // Convert flattened index to matrix coordinates
    int row = (int)((-1 + sqrt(8 * idx + 1)) / 2);
    int col = idx - row * (row + 1) / 2;
    
    float sum = 0.0f;
    for (int k = col; k <= row; ++k) {
        sum += A[row * N + k] * B[k * N + col];
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
    auto C = torch::zeros_like(A);

    const int total_elements = N * (N + 1) / 2;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    warp_uniform_combined_kernel<<<grid_size, block_size>>>(
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
    m.def("forward", &forward, "Combined warp-uniform and tiled triangular matrix multiplication (CUDA)");
}