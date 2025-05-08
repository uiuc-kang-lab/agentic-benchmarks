#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using warp-level primitives for optimization
__global__ void diag_matmul_warp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * M;
    if (idx < total) {
        int row = idx / M;
        int col = idx % M;
        float a_val = A[row];
        float b_val = B[row * M + col];

        // Use warp shuffle to optimize small reductions
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            a_val += __shfl_down_sync(0xFFFFFFFF, a_val, offset);
            b_val += __shfl_down_sync(0xFFFFFFFF, b_val, offset);
        }

        // Only one thread in the warp writes the result
        if (threadIdx.x % warpSize == 0) {
            C[idx] = a_val * b_val;
        }
    }
}

// Forward function that wraps our CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are on contiguous memory
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create an output tensor with the same device and type as B
    auto C = torch::empty({N, M}, B.options());

    // Configure and launch the kernel
    const int64_t threads = 256;
    const int64_t blocks = (N * M + threads - 1) / threads;
    diag_matmul_warp_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Create the PyTorch extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication of A and B on the GPU");
}