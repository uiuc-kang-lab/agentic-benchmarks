#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// CUDA kernel: Each warp processes one or more rows using stride loops for both rows and columns.
// For each row, lane 0 loads the diagonal element from A, which is then broadcast to all lanes using __shfl_sync.
// Threads iterate over columns in a stride loop to cover all columns even if the workload exceeds the number of threads.
__global__ void diag_matmul_kernel_stride(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int totalWarps = (gridDim.x * blockDim.x) / WARP_SIZE;

    // Loop over rows assigned to each warp using a stride loop
    for (int row = warpId; row < N; row += totalWarps) {
        float a_val;
        // Lane 0 loads the diagonal element
        if (lane == 0) {
            a_val = A[row];
        }
        // Broadcast the diagonal element to all lanes within the warp
        a_val = __shfl_sync(0xffffffff, a_val, 0);

        int row_offset = row * M;
        // Loop over columns in a stride manner
        for (int col = lane; col < M; col += WARP_SIZE) {
            int idx = row_offset + col;
            C[idx] = a_val * B[idx];
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

    // Configure kernel: use a stride loop that covers all rows if there are more rows than available warps.
    const int threadsPerBlock = 128;  // Must be a multiple of WARP_SIZE
    const int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    // Calculate the number of blocks such that each block provides a number of warps that may cover N rows with striding
    int blocks = (N + warpsPerBlock - 1) / warpsPerBlock;

    diag_matmul_kernel_stride<<<blocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with stride loops for workloads larger than available threads");
}
