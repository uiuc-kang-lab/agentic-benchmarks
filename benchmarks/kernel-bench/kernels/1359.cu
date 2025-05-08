#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each block processes one row of B. The diagonal element from A is loaded once into shared memory.
__global__ void diag_matmul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x; // Each block handles one row
    __shared__ float a_val;
    
    // Load the diagonal element for the current row into shared memory
    if (threadIdx.x == 0) {
        a_val = A[row];
    }
    __syncthreads(); // Synchronize to ensure a_val is available to all threads in the block
    
    // Compute output for the given row; each thread covers multiple columns if needed
    int col = threadIdx.x;
    while (col < M) {
        C[row * M + col] = a_val * B[row * M + col];
        col += blockDim.x;
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

    // Create CUDA streams for overlapping computation and memory transfers
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch one block per row with a fixed number of threads per block
    const int threads = 256;
    diag_matmul_shared_kernel<<<N, threads, 0, stream1>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    // Synchronize streams and clean up
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matrix multiplication using shared memory and CUDA streams");
}