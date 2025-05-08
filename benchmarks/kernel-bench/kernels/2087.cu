#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute the lower triangular part of C = A * B for lower triangular matrices.
// It uses a grid-stride loop over exactly N*(N+1)/2 elements (the valid lower-triangular entries) to ensure
// that every thread does useful work, thereby better balancing the workload.
__global__ void triangular_lower_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N,
                                          int total_lower) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < total_lower; idx += blockDim.x * gridDim.x) {
        // Map linear index 'idx' to matrix coordinates (row, col) for the lower triangular region.
        // The relationship is: idx = row*(row+1)/2 + col, with 0 <= col <= row.
        float temp = sqrtf(8.0f * idx + 1.0f);
        int row = (int)floorf((temp - 1.0f) * 0.5f);
        int start = row * (row + 1) / 2;
        int col = idx - start;
        
        // Compute C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col]
        float sum = 0.0f;
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel to fill the upper triangular part (where row < col) of C with zeros.
// This ensures the final result is a full matrix with zero entries in the upper triangular region.
__global__ void fill_upper_kernel(float* __restrict__ C, int N, int total_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int row = idx / N;
        int col = idx % N;
        if (row < col) {
            C[row * N + col] = 0.0f;
        }
    }
}

// C++ interface exposed to PyTorch. It launches two kernels:
// 1. triangular_lower_kernel: computes the lower triangular part in a load-balanced manner using a grid-stride loop.
// 2. fill_upper_kernel: fills the remaining upper triangular part with zeros.

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

    // Total number of valid lower triangular elements
    int total_lower = N * (N + 1) / 2;
    int total_elements = N * N;

    // Choose a block size and compute grid sizes for both kernels
    int block_size = 256;
    int num_blocks_lower = (total_lower + block_size - 1) / block_size;
    int num_blocks_full = (total_elements + block_size - 1) / block_size;

    // Launch kernel to compute the lower triangular part
    triangular_lower_kernel<<<num_blocks_lower, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        total_lower
    );

    // Launch kernel to fill the upper triangular region with zeros
    fill_upper_kernel<<<num_blocks_full, block_size>>>(
        C.data_ptr<float>(),
        N,
        total_elements
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Evenly distributed triangular matrix multiplication (CUDA)");
}
