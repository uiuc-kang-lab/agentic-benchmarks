#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define maximum number of floats in constant memory. This limits matrix dimensions to 128x128 (16384 floats).
#define MAX_SIZE 16384

// Declare constant memory arrays for read-only matrices A and B.
__constant__ float constA[MAX_SIZE];
__constant__ float constB[MAX_SIZE];

// CUDA kernel that uses constant memory for matrices A and B.
// It computes the lower-triangular matrix multiplication: C = tril(A * B).
__global__ void constant_memory_kernel(float* __restrict__ C, int N, int start_row, int end_row) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y + start_row;

    if (row >= end_row || col >= N) return;

    float sum = 0.0f;
    // Only compute for lower triangular part
    for (int k = col; k <= row; ++k) {
        // Access A and B from constant memory
        sum += constA[row * N + k] * constB[k * N + col];
    }

    C[row * N + col] = (row < col) ? 0.0f : sum;
}

// Forward function exposed to PyTorch
at::Tensor forward_constant_optimized(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    int N = A.size(0);
    TORCH_CHECK(A.size(0) == A.size(1), "Matrix A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "Matrix B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");
    TORCH_CHECK(N * N <= MAX_SIZE, "Matrix size exceeds constant memory capacity");

    // Copy matrices A and B into constant memory
    cudaMemcpyToSymbol(constA, A.data_ptr<float>(), N * N * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(constB, B.data_ptr<float>(), N * N * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    auto C = torch::empty_like(A);

    // Configure block and grid dimensions
    const dim3 threadsPerBlock(32, 32);
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, N);
        if (start >= end) continue;
        
        dim3 blocks(
            (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (end - start + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        
        constant_memory_kernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(
            C.data_ptr<float>(), N, start, end
        );
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_constant_optimized, "Constant memory optimized triangular matmul (CUDA)");
}
