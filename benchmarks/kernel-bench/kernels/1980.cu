#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimize memory coalescing by transposing thread map for column-contiguous B accesses
__global__ void coalesced_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Map thread.x to column dimension for coalesced B[k][col] accesses
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
    } else {
        float sum = 0.0f;
        // A access remains row-wise (coalesced through L1 cache)
        // B access becomes column-wise but with thread.x = col -> coalesced
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Use 32x8 blocks to keep 4 active warps per SM
    // and maximize memory coalescing (x-dim stays dense for B accesses)
    const int bx = 32, by = 8;
    dim3 threads(bx, by);
    dim3 blocks((N + bx - 1) / bx, (N + by - 1) / by);

    coalesced_triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced accesses triangular matmul (CUDA)");
}