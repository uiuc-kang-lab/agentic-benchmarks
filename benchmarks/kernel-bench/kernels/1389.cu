#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using a flattened grid with stride loops to handle all elements
__global__ void diag_matmul_stride_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Total number of elements in the output matrix
    int64_t total = N * M;
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Stride across the grid
    int stride = gridDim.x * blockDim.x;
    
    // Loop over all elements assigned to this thread
    for (int i = idx; i < total; i += stride) {
        int row = i / M; // Compute row index
        C[i] = A[row] * B[i];
    }
}

// Forward function wrapping the CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");
    
    // Ensure tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create output tensor
    auto C = torch::empty({N, M}, B.options());

    // Define thread/block configuration
    int threads = 32;
    int blocks = (N * M + threads - 1) / threads;

    diag_matmul_stride_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
    
    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using a flattened stride loop");
}
