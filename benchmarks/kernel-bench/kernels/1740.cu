#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int T = N * (N + 1) / 2;
    if (k >= T) return;

    // Fast row calculation with sqrt
    float fk = static_cast<float>(k);
    float sqrt_val = sqrtf(8.0f * fk + 1.0f);
    int row = static_cast<int>((sqrt_val - 1.0f) * 0.5f);

    // Adjust row boundaries
    while ((row + 1) * (row + 2) / 2 <= k) row++;
    while (row * (row + 1) / 2 > k) row--;

    int col = k - (row * (row + 1)) / 2;

    float sum = 0.0f;
    for(int k_idx = col; k_idx <= row; ++k_idx) {
        sum += A[row * N + k_idx] * B[k_idx * N + col];
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
    auto C = torch::empty_like(A);

    int T = N * (N + 1) / 2;
    int block_size = 128;
    int grid_size = (T + block_size - 1) / block_size;

    optimized_triangular_mm_kernel<<<grid_size, block_size>>>(
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
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}