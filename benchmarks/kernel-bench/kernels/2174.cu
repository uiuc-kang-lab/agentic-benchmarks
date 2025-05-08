#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel that computes C = A.T * B using warp-level primitives for reduction.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// Note: Each output element C[i,j] = sum_{k=0}^{K-1} A[k * M + i] * B[k * N + j]
// We assign one warp per output element and partition the K dimension among lanes.
__global__ void matMulKernelWarp(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int K,
                                 int M,
                                 int N) {
    // Each block is organized as a 2D grid of threads with blockDim.x = 32 (warp size) and blockDim.y = number of warps per block.
    // Compute global warp id. Since each warp (i.e. each row in the block) computes one output element:
    int warpsPerBlock = blockDim.y;  // because blockDim.x is fixed to warp size (32)
    int globalWarpId = blockIdx.x * warpsPerBlock + threadIdx.y;

    int totalElements = M * N;
    if (globalWarpId >= totalElements) {
        return;
    }
    
    // Map linear index to 2D indices: i for row of C (and column of A), j for column of C (and B)
    int i = globalWarpId / N;
    int j = globalWarpId % N;

    // Each warp has 32 threads. Lane id is threadIdx.x (0...31).
    int lane = threadIdx.x;
    float sum = 0.0f;

    // Partition the K dimension among lanes in the warp.
    for (int k = lane; k < K; k += 32) {
        // A is stored as (K, M): A[k, i] is at A[k * M + i]
        // B is stored as (K, N): B[k, j] is at B[k * N + j]
        sum += A[k * M + i] * B[k * N + j];
    }

    // Perform warp-level reduction using __shfl_down_sync without shared memory.
    // Use full mask for active threads: 0xffffffff.
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first lane writes the result to global memory.
    if (lane == 0) {
        C[i * N + j] = sum;
    }
}

// The forward function is exposed via PyBind11 and can be called from Python.
// A: Tensor of shape (K, M) [CUDA, float32]
// B: Tensor of shape (K, N) [CUDA, float32]
// Returns: Tensor C of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of type float32.
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Dimensions: A: (K, M), B: (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Configure kernel launch parameters.
    // We assign one warp per output element. Using blockDim.x = 32 (warp size) and blockDim.y, e.g., 8 warps per block.
    dim3 blockDim(32, 8);
    int warpsPerBlock = blockDim.y;  // 8 warps per block
    int totalWarps = M * N;          // one warp computes one element of C
    int gridSize = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the kernel.
    matMulKernelWarp<<<gridSize, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with warp-level reduction (CUDA)");
}
