#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for diagonal matrix A
__constant__ float const_A[1024];  // Assuming a maximum size of 1024 for demonstration

// CUDA kernel: each thread computes one element of the output C
// C[i, j] = A[i] * B[i, j] where A is diagonal
__global__ void diag_matmul_kernel(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Calculate 2D thread index
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one row
    if (row < N) {
        // Get the diagonal element for this row
        const float diag_elem = const_A[row];
        
        // Multiply each element in the row by the diagonal element
        for (int col = 0; col < M; col++) {
            C[row * M + col] = diag_elem * B[row * M + col];
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

    // Copy diagonal matrix A to constant memory
    cudaMemcpyToSymbol(const_A, A.data_ptr<float>(), N * sizeof(float));

    // Create an output tensor with the same device and type as B
    auto C = torch::empty({N, M}, B.options());

    // Configure and launch the kernel
    const int64_t threads = 256;
    const int64_t blocks = (N * M + threads - 1) / threads;
    diag_matmul_kernel<<<blocks, threads>>>(
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