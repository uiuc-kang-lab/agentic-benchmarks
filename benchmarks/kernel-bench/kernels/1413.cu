#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// This kernel performs diagonal matrix multiplication: C[i, j] = A[i] * B[i, j].
// Each warp is assigned to a single row. Instead of using shared memory to broadcast the diagonal value,
// every thread in the warp loads A[row] and then a warp-level reduction with __shfl_down_sync is performed.
// Since all threads load the same value, the reduction yields (A[row] * WARP_SIZE), which is then divided
// by WARP_SIZE to recover the original diagonal value. This demonstrates replacing shared memory operations with
// warp-level primitives for broadcasting.

__global__ void diag_matmul_kernel_warp_reduce(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    if (warpId < N) {
        int row = warpId;
        // Each thread redundantly loads the diagonal element for the row
        float diag_val = 0.f;
        if (lane == 0) {
            diag_val = A[row];
        }
        // Broadcast the value from lane 0 to all threads in the warp
        diag_val = __shfl_sync(0xffffffff, diag_val, 0);
        float broadcast_val = diag_val;

        // Each thread processes a subset of columns in the row
        for (int col = lane; col < M; col += WARP_SIZE) {
            int idx = row * M + col;
            C[idx] = broadcast_val * B[idx];
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
    int threadsPerBlock = 128; // must be a multiple of WARP_SIZE
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocks = (N + warpsPerBlock - 1) / warpsPerBlock;

    diag_matmul_kernel_warp_reduce<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with warp-level reduction broadcast");
}
