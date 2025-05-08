#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses shared memory and minimizes __syncthreads() calls
__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    extern __shared__ float shared_A[];
    int row = blockIdx.x;
    int col = threadIdx.x + blockDim.x * blockIdx.y;

    if (row < N && col < M) {
        if (threadIdx.x == 0) {
            shared_A[threadIdx.y] = A[row];
        }
        __syncthreads();

        float a_val = shared_A[threadIdx.y];
        C[row * M + col] = a_val * B[row * M + col];
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    const int threads = 128;
    dim3 blocks(N, (M + threads - 1) / threads);

    size_t shared_mem_size = threads * sizeof(float);
    diag_matmul_kernel<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matrix multiplication with minimal synchronization");
}