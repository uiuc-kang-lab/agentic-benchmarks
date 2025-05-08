#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Compute lower triangular elements using a 1D flattened index.
// For each thread, we map a global thread id 'tid' in [0, N*(N+1)/2) to a matrix element (row, col)
// in the lower triangle, using the formula: tid = row*(row+1)/2 + col.
// Each thread then computes: C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col].

__global__ void triangular_mm_kernel_lower(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N,
                                             int lowerCount) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < lowerCount) {
        // Compute 'row' using the inverse triangular number formula.
        // row = floor((sqrt(8*tid + 1) - 1) / 2)
        float tid_f = (float) tid;
        int row = (int)((sqrtf(8.0f * tid_f + 1.0f) - 1.0f) * 0.5f);
        int rowStart = row * (row + 1) / 2;
        int col = tid - rowStart;

        // Compute the dot-product for the lower triangular element.
        float sum = 0.0f;
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel 2: Fill the upper triangular part of C with zeros.
// This kernel launches over all N*N elements, and for each (row, col) where row < col, sets C[row, col] = 0.

__global__ void fill_upper_kernel(float* __restrict__ C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        int row = idx / N;
        int col = idx % N;
        if (row < col) {
            C[idx] = 0.0f;
        }
    }
}

// The forward function sets up the kernels and launches them in sequence.
// First, we compute the lower triangular part with a 1D kernel for even workload distribution,
// then we launch a second kernel to fill the upper triangular part with zeros.

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

    // Total number of lower triangular elements
    int lowerCount = N * (N + 1) / 2;

    // Configure 1D grid for the lower triangular kernel
    int threads = 256;
    int blocks_lower = (lowerCount + threads - 1) / threads;
    triangular_mm_kernel_lower<<<blocks_lower, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        lowerCount
    );

    // Configure 1D grid for the upper triangular fill kernel
    int totalElements = N * N;
    int blocks_upper = (totalElements + threads - 1) / threads;
    fill_upper_kernel<<<blocks_upper, threads>>>(C.data_ptr<float>(), N);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Evenly distributed triangular matrix multiplication (CUDA)");
}
