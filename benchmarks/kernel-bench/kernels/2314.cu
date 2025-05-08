#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// You can experiment with different block sizes: 32, 64, 128, 256, 512
// Here, BLOCK_SIZE should be a multiple of 32 (the warp size).
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Kernel: each warp (32 threads) computes one output element of matrix C.
// For each output element C[m][n], the warp computes the dot product of row m of A and row n of B (since B is stored as transposed input).
__global__ void matmul_transposed_b_block_size_kernel(const float* __restrict__ A,
                                                        const float* __restrict__ B,
                                                        float* __restrict__ C,
                                                        int M, int N, int K) {
    const int warpSize = 32;
    // Compute number of warps per block
    int warpsPerBlock = blockDim.x / warpSize;
    // Identify this thread's warp ID within the block
    int warpId = threadIdx.x / warpSize;
    // Compute global warp ID
    int globalWarpId = blockIdx.x * warpsPerBlock + warpId;

    // Each warp computes one output element. Map global warp id to element index in C.
    int elem_id = globalWarpId;
    if (elem_id < M * N) {
        int m = elem_id / N;  // row index in C (and A)
        int n = elem_id % N;  // column index in C (and corresponds to row index in B since B is transposed)

        int lane = threadIdx.x % warpSize;  // lane index within the warp
        float sum = 0.0f;
        // Each lane processes a subset of the K dimension with stride = warpSize
        for (int k = lane; k < K; k += warpSize) {
            // Use __ldg to leverage the read-only data cache
            float a_val = __ldg(&A[m * K + k]);
            float b_val = __ldg(&B[n * K + k]);
            sum += a_val * b_val;
        }
        
        // Warp-level reduction using __shfl_down_sync
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First lane writes the result
        if (lane == 0) {
            C[m * N + n] = sum;
        }
    }
}


// Host function: determines grid and block configurations based on M, N, K and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Each warp computes one output element. Total elements in C = M * N.
    // Number of warps per block is BLOCK_SIZE / 32.
    int warpsPerBlock = BLOCK_SIZE / 32;
    int totalWarpsNeeded = M * N; // one warp per output element
    int blocks = (totalWarpsNeeded + warpsPerBlock - 1) / warpsPerBlock;

    // Launch the kernel with the chosen block size. You can experiment with different BLOCK_SIZE values.
    matmul_transposed_b_block_size_kernel<<<blocks, BLOCK_SIZE>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using configurable block sizes (CUDA)");
}
