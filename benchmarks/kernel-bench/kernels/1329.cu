#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void warp_optimized_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    if (row < N) {
        float a_val = A[row];

        // Use warp shuffle to broadcast a_val to all threads in the warp
        a_val = __shfl_sync(0xFFFFFFFF, a_val, 0);

        // Each warp processes a row
        for (int j = col; j < M; j += blockDim.x) {
            int idx = row * M + j;
            C[idx] = a_val * B[idx];
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    const int threads_x = 32;  // Number of threads per warp
    const int threads_y = 8;   // Number of warps per block
    dim3 threads(threads_x, threads_y);
    dim3 blocks((N + threads_y - 1) / threads_y);

    warp_optimized_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized diagonal matrix multiplication");
}