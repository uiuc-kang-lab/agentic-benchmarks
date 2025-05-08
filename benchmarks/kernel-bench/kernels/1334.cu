#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses shared memory for computation
__global__ void shared_memory_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    extern __shared__ float shared_B[];
    int tx = threadIdx.x;
    int row = blockIdx.x;
    float a_val = A[row];
    int offset = row * M;

    // Load data into shared memory
    for (int j = tx; j < M; j += blockDim.x) {
        shared_B[j] = B[offset + j];
    }
    __syncthreads();

    // Compute the matrix multiplication
    for (int j = tx; j < M; j += blockDim.x) {
        C[offset + j] = a_val * shared_B[j];
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

    int threads = 256;
    size_t shared_mem_size = M * sizeof(float);

    shared_memory_diag_matmul_kernel<<<N, threads, shared_mem_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using shared memory");
}
