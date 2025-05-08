#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Kernel implementing split-K tiled matrix multiplication with minimal atomic operations
// When split_k > 1, the K-dimension workload is partitioned among blocks (gridDim.z), and partial results are
// accumulated using atomicAdd. If split_k == 1, no atomics are used since each output element is computed uniquely.

__global__ void SplitKMatmulKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N, int split_k) {
    // Compute row and column indices for the output matrix
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // gridDim.z represents the split-K factor; each block in z dimension processes a partition of K
    int split_id = blockIdx.z;

    float sum = 0.0f;
    // Compute the chunk size for the K-dimension partition
    int chunk = (K + split_k - 1) / split_k;
    int k_start = split_id * chunk;
    int k_end = k_start + chunk;
    if (k_end > K) k_end = K;

    // Compute the partial dot product over the assigned K-partition
    for (int k = k_start; k < k_end; k++) {
        if (row < M && k < K && col < N) {
            sum += A[row * K + k] * B[k * N + col];
        }
    }

    // Write the computed partial result into C
    if (row < M && col < N) {
        if (split_k == 1) {
            // If only one partition, write directly
            C[row * N + col] = sum;
        } else {
            // If multiple partitions, accumulate using atomicAdd to prevent race conditions
            atomicAdd(&C[row * N + col], sum);
        }
    }
}

// Forward function: binds to Python, checks inputs, allocates output tensor, and launches the kernel
// The split_k parameter allows for additional parallelism along the K dimension. For a small K, split_k may be 1
// to avoid contention from atomic operations.

torch::Tensor forward(torch::Tensor A, torch::Tensor B, int split_k=1) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor and initialize to zero. When split_k > 1, accumulation via atomics is required
    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions. gridDim.z is set to split_k to partition the K dimension workload
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH,
                 split_k);

    SplitKMatmulKernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N, split_k);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Split-K Tiled Matrix Multiplication with minimal atomic operations (CUDA)");
}
