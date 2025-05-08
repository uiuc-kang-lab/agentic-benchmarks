#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32

__global__ void optimized_divergence_kernel(const float* __restrict__ A,
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
    TORCH_CHECK(A.size(0) == B.size(0), "Matrix dimension mismatch");
    const int N = A.size(0);
    auto C = torch::zeros_like(A);

    const int total_elements = N * (N + 1) / 2;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    optimized_divergence_kernel<<<grid_size, block_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-uniform triangular mm");
}