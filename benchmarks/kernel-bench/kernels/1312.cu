#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel: each block handles one entire row of B.
// This avoids divergent branching across rows and ensures uniform control flow within warps.
__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x;  // one block per row
    if (row < N) {
        float a_val = A[row];
        
        // Compute the largest multiple of blockDim.x that is <= M
        int main_end = (M / blockDim.x) * blockDim.x;
        
        // Main loop: All threads execute uniformly on the main chunk
        for (int j = threadIdx.x; j < main_end; j += blockDim.x) {
            int idx = row * M + j;
            C[idx] = a_val * B[idx];
        }
        
        // Tail loop: handles any remaining columns with a minimal if check
        for (int j = main_end + threadIdx.x; j < M; j += blockDim.x) {
            int idx = row * M + j;
            C[idx] = a_val * B[idx];
        }
    }
}

// Forward function wrapping our optimized CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Launch one block per row, with threads chosen as a multiple of warp size.
    int threads = (M > 256) ? 256 : (((M + 31) / 32) * 32);
    dim3 grid(N);
    diag_matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication optimized for warp divergence");
}
