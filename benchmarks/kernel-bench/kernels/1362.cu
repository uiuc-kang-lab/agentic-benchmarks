#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Optimized grid size to better cover the data domain
__global__ void diag_matmul_shared_coalesced_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Calculate global thread index and stride
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Calculate total elements to process
    const int64_t total_elements = N * M;
    
    // Process elements in strided fashion
    for (int64_t idx = tid; idx < total_elements; idx += stride) {
        // Calculate row and column from linear index
        const int64_t row = idx / M;
        const int64_t col = idx % M;
        
        // Each thread reads its needed diagonal element directly
        // Using __ldg for cached read of A since it's read multiple times
        const float a_val = __ldg(&A[row]);
        const float b_val = __ldg(&B[row * M + col]);
        C[row * M + col] = a_val * b_val;
    }
}

// Forward function wraps the CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create output tensor with the same type and device as B
    auto C = torch::empty({N, M}, B.options());

    // Launch one block per row with a fixed number of threads per block
    const int threads = 256;
    diag_matmul_shared_coalesced_kernel<<<N, threads>>>(
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
    m.def("forward", &forward, "Diagonal matrix multiplication with coalesced memory access");
}