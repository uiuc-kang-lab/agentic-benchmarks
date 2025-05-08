#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes C = A * B^T using warp-level primitives and loop unrolling.
// Each warp computes one output element C[i, j] using registers and fast __shfl_down_sync reduction.

__global__ void warp_matmul_transposed_kernel_unroll(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int M, int N, int K) {
    // Define warp size
    const unsigned int warpSize = 32;
    // Get lane index within warp and warp id within block
    unsigned int lane = threadIdx.x; // range [0, 31]
    unsigned int warpId = threadIdx.y; // each block can have several warps in y dimension

    // Map each warp to one output element
    int row = blockIdx.y * blockDim.y + warpId;  // index into A (row)
    int col = blockIdx.x;                          // index into B (treated as row due to transposition)

    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Loop unrolling by a factor of 8: each thread processes 8 elements per iteration
        // This better aligns with warp boundaries (8 * 32 = 256 elements per iteration)
        int k;
        for (k = lane; k <= K - int(warpSize * 8); k += warpSize * 8) {
            float a0 = A[row * K + k];
            float a1 = A[row * K + k + warpSize];
            float a2 = A[row * K + k + 2 * warpSize];
            float a3 = A[row * K + k + 3 * warpSize];
            float a4 = A[row * K + k + 4 * warpSize];
            float a5 = A[row * K + k + 5 * warpSize];
            float a6 = A[row * K + k + 6 * warpSize];
            float a7 = A[row * K + k + 7 * warpSize];
            
            float b0 = B[col * K + k];
            float b1 = B[col * K + k + warpSize];
            float b2 = B[col * K + k + 2 * warpSize];
            float b3 = B[col * K + k + 3 * warpSize];
            float b4 = B[col * K + k + 4 * warpSize];
            float b5 = B[col * K + k + 5 * warpSize];
            float b6 = B[col * K + k + 6 * warpSize];
            float b7 = B[col * K + k + 7 * warpSize];
            
            sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 +
                  a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7;
        }
        // Process any remaining elements
        for (; k < K; k += warpSize) {
            sum += A[row * K + k] * B[col * K + k];
        }

        // Warp-level reduction using __shfl_down_sync
        // All 32 threads in the warp participate
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // The first thread in the warp writes the result
        if (lane == 0) {
            C[row * N + col] = sum;
        }
    }
}

// Forward function called from PyTorch
// It launches one warp per output element. 
// Block configuration: blockDim.x = 32 (warp size), blockDim.y = warpsPerBlock (e.g., 8).
// Grid configuration: grid.x covers output columns (N) and grid.y covers groups of output rows.

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

    // Configure launch parameters
    const int warpSize = 32;
    const int warpsPerBlock = 8; // adjust this based on occupancy
    dim3 block(warpSize, warpsPerBlock);
    dim3 grid(N, (M + warpsPerBlock - 1) / warpsPerBlock);

    warp_matmul_transposed_kernel_unroll<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level unrolled matrix multiplication with transposed B (CUDA)");
}
