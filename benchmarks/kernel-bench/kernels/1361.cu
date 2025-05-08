#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with minimal synchronization and vectorized memory access where possible
__global__ void diag_matmul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x;
    
    // Use shared memory for the diagonal element
    __shared__ float a_val;
    
    // Only one thread loads the diagonal element
    if (threadIdx.x == 0) {
        a_val = A[row];
    }
    
    // Single synchronization point - necessary for shared memory consistency
    __syncthreads();
    
    // Process four elements at a time when possible
    int col = threadIdx.x;
    const int stride = blockDim.x;
    const float4* B_vec = reinterpret_cast<const float4*>(B + row * M);
    float4* C_vec = reinterpret_cast<float4*>(C + row * M);
    
    // Handle vectorized loads first
    const int vec_limit = M / 4;
    while (col < vec_limit) {
        float4 b4 = B_vec[col];
        float4 c4;
        c4.x = a_val * b4.x;
        c4.y = a_val * b4.y;
        c4.z = a_val * b4.z;
        c4.w = a_val * b4.w;
        C_vec[col] = c4;
        col += stride;
    }
    
    // Handle remaining elements
    col = threadIdx.x + (vec_limit * 4);
    while (col < M) {
        C[row * M + col] = a_val * B[row * M + col];
        col += stride;
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

    // Create output tensor
    auto C = torch::empty({N, M}, B.options());

    // Launch kernel with one block per row
    const int threads = 256;
    diag_matmul_shared_kernel<<<N, threads>>>(
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
    m.def("forward", &forward, "Optimized diagonal matrix multiplication with minimal synchronization");
}