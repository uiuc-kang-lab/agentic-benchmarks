#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_DIAG_SIZE 16384

// Store the diagonal matrix in constant memory for fast read-only access
__constant__ float d_A[MAX_DIAG_SIZE];

// CUDA kernel: each warp processes one row of B using the diagonal element from constant memory
__global__ void diag_matmul_kernel_warp_const(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Determine warp lane and warp index
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    // Each warp processes one row if within bounds
    if (warpId < N) {
        // Lane 0 loads the diagonal element from constant memory
        float a_val;
        if (lane == 0) {
            a_val = d_A[warpId];
        }
        // Broadcast the value to all lanes in the warp
        a_val = __shfl_sync(0xffffffff, a_val, 0);

        int row = warpId;
        // Process columns in increments of WARP_SIZE
        for (int col = lane; col < M; col += WARP_SIZE) {
            int idx = row * M + col;
            C[idx] = a_val * B[idx];
        }
    }
}

// Forward function wrapping the CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");
    TORCH_CHECK(A.size(0) <= MAX_DIAG_SIZE, "Diagonal matrix size exceeds the constant memory limit");

    // Ensure inputs are contiguous on the GPU
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Copy diagonal data into constant memory
    cudaMemcpyToSymbol(d_A, A.data_ptr<float>(), N * sizeof(float));

    // Create the output tensor with the same options as B
    auto C = torch::empty({N, M}, B.options());

    // Configure kernel: one warp per row
    int threadsPerBlock = 256; // Increased to improve occupancy
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int totalWarpsNeeded = N; // One warp per row
    int blocks = (totalWarpsNeeded + warpsPerBlock - 1) / warpsPerBlock;

    diag_matmul_kernel_warp_const<<<blocks, threadsPerBlock>>>(
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using warp-level broadcast and constant memory");
}
