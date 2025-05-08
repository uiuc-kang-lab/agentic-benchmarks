#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Compute the lower triangular part of C = A * B using a flattened 1D grid
// Each thread computes one element in the lower triangle, where the flat index maps to (row, col)
// using the relation: idx = row*(row+1)/2 + col, with col in [0, row].
__global__ void triangular_mm_lower_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N,
                                            int LT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= LT) return;

    // Map flat index idx to (row, col) in the lower triangular matrix
    float f = sqrtf(8.f * idx + 1.f);
    int row = (int)((f - 1.f) * 0.5f);
    int rowStart = row * (row + 1) / 2;
    // Adjust for any rounding error
    while ((row + 1) * (row + 2) / 2 <= idx) {
        row++;
        rowStart = row * (row + 1) / 2;
    }
    int col = idx - rowStart;

    float sum = 0.f;
    // Compute the inner product from k = col to row
    for (int k = col; k <= row; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Kernel 2: Fill the upper triangular part of C with zeros
__global__ void fill_upper_kernel(float* __restrict__ C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx >= total) return;
    int row = idx / N;
    int col = idx % N;
    if (row < col) {
        C[row * N + col] = 0.f;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N_int = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    // Number of elements in the lower triangular part
    int LT = N_int * (N_int + 1) / 2;
    const int blockSize = 256;
    int gridSizeLower = (LT + blockSize - 1) / blockSize;

    // Launch kernel to compute lower triangular elements
    triangular_mm_lower_kernel<<<gridSizeLower, blockSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N_int,
        LT
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "triangular_mm_lower_kernel launch failed: ", cudaGetErrorString(err));

    // Launch a second kernel to fill the upper triangular part with zeros
    int totalElements = N_int * N_int;
    int gridSizeUpper = (totalElements + blockSize - 1) / blockSize;
    fill_upper_kernel<<<gridSizeUpper, blockSize>>>(C.data_ptr<float>(), N_int);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fill_upper_kernel launch failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Distributed workload triangular matmul (CUDA)");
}
