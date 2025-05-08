#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: optimized for memory coalescing
__global__ void diag_matmul_kernel_coalesced(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Calculate global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple elements to reduce register pressure
    // and increase arithmetic intensity
    const int stride = gridDim.x * blockDim.x;
    
    for (int row = tid; row < N; row += stride) {
        // Cache the diagonal element to reduce global memory access
        const float a_val = A[row];
        
        // Process elements in chunks to better utilize cache
        const int chunk_size = 4;
        const int row_offset = row * M;
        
        // Process main chunks
        int col = 0;
        for (; col <= M - chunk_size; col += chunk_size) {
            const int idx = row_offset + col;
            // Vectorized multiplication for better instruction throughput
            C[idx] = a_val * B[idx];
            C[idx + 1] = a_val * B[idx + 1];
            C[idx + 2] = a_val * B[idx + 2];
            C[idx + 3] = a_val * B[idx + 3];
        }
        
        // Handle remaining elements
        for (; col < M; ++col) {
            const int idx = row_offset + col;
            C[idx] = a_val * B[idx];
        }
    }
}

// Forward function that wraps our CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are on contiguous memory
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create an output tensor with the same device and type as B
    auto C = torch::empty({N, M}, B.options());

    // Configure and launch the kernel
    const int64_t threads = 256;
    const int64_t blocks = (N + threads - 1) / threads;
    diag_matmul_kernel_coalesced<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Create the PyTorch extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication of A and B on the GPU");
}