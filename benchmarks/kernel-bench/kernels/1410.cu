#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// CUDA kernel: each warp processes one row of B, broadcasting the diagonal value from A
__global__ void diag_matmul_kernel_warp_broadcast_min_divergence(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Compute warp-local lane and global warp index
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    // Each warp processes one row if within bounds
    if (warpId < N) {
        // Lane 0 loads the diagonal element and then broadcasts it
        float a_val = __shfl_sync(0xffffffff, A[warpId], 0);

        const int row = warpId;
        const int row_offset = row * M;

        // Use vectorized memory operations if M is divisible by 4 for better performance
        if (M % 4 == 0) {
            int vec_M = M / 4;
            for (int vec = lane; vec < vec_M; vec += WARP_SIZE) {
                int idx = row * vec_M + vec;
                float4 b_val = reinterpret_cast<const float4*>(B)[idx];
                float4 c_val;
                c_val.x = a_val * b_val.x;
                c_val.y = a_val * b_val.y;
                c_val.z = a_val * b_val.z;
                c_val.w = a_val * b_val.w;
                reinterpret_cast<float4*>(C)[idx] = c_val;
            }
        } else {
            // Fallback to scalar operations if M is not divisible by 4
            for (int col = lane; col < M; col += WARP_SIZE) {
                int idx = row_offset + col;
                C[idx] = a_val * B[idx];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    // Configure kernel: assign one warp per row
    int threadsPerBlock = 128; // Must be multiple of 32
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int totalWarpsNeeded = N; // one warp per row
    int blocks = (totalWarpsNeeded + warpsPerBlock - 1) / warpsPerBlock;

    diag_matmul_kernel_warp_broadcast_min_divergence<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using warp-level broadcast with minimized divergence");
}