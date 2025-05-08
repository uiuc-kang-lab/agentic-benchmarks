#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for computing C = A.T * B with minimized warp divergence
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
__global__ void coalescedMatMulKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int K,
                                     const int M,
                                     const int N) {
    // Use constant size tiles for better compiler optimization
    constexpr int TILE_SIZE = 32;  // Match warp size
    
    // Calculate global thread position
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Early exit if this thread is outside matrix bounds
    // This check is done once per thread, avoiding repeated boundary checks
    if (row >= M || col >= N) return;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Process K dimension in tiles
    #pragma unroll 4
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        const int k_start = tile * TILE_SIZE;
        const int k_end = min(k_start + TILE_SIZE, K);
        
        // Main computation loop - no divergent branches inside
        #pragma unroll
        for (int k = k_start; k < k_end; k++) {
            sum += A[k * M + row] * B[k * N + col];
        }
    }
    
    // Store result
    C[row * N + col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    const int K = A.size(0);
    const int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Configure kernel launch parameters
    constexpr int TILE_SIZE = 32;  // Match warp size for optimal execution
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE/8);  // 32x4 threads per block
    dim3 numBlocks(
        (M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch kernel with computed grid configuration
    coalescedMatMulKernel<<<numBlocks, threadsPerBlock>>>(
        A_ptr, B_ptr, C_ptr, K, M, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with coalesced memory access (CUDA)");
}