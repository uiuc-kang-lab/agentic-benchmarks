#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel: each thread computes one element of the output C
// C[i, j] = A[i] * B[i, j]
__global__ void optimized_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

// Forward function that wraps our optimized CUDA kernel
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

    // Create CUDA streams for concurrent execution
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Split the work into two parts along the N dimension
    int64_t N_half = N / 2;
    
    // Configure thread blocks and grid dimensions for each half
    dim3 threads(16, 16);
    dim3 blocks1((N_half + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    dim3 blocks2(((N - N_half) + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    // Launch kernels in separate streams
    optimized_diag_matmul_kernel<<<blocks1, threads, 0, stream1>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N_half,
        M
    );

    optimized_diag_matmul_kernel<<<blocks2, threads, 0, stream2>>>(
        A.data_ptr<float>() + N_half,
        B.data_ptr<float>() + (N_half * M),
        C.data_ptr<float>() + (N_half * M),
        N - N_half,
        M
    );

    // Synchronize streams and clean up
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return C;
}

// Create the PyTorch extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matrix multiplication of A and B on the GPU");
}