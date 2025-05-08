#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each block processes one row. Instead of using shared memory and synchronization, each thread reads
// the diagonal element directly from global memory using __ldg to leverage the read-only cache. This avoids
// unnecessary atomics since no race conditions exist, as each thread writes to a unique output element.

__global__ void diag_matmul_readonly_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x;
    // Each thread loads the diagonal element independently using the read-only cache
    float a = __ldg(&A[row]);

    int thread_id = threadIdx.x;
    int stride = blockDim.x;

    // Process four elements at a time when possible
    int vec_limit = M / 4;
    const float4* B_vec = reinterpret_cast<const float4*>(B + row * M);
    float4* C_vec = reinterpret_cast<float4*>(C + row * M);

    for (int col = thread_id; col < vec_limit; col += stride) {
        float4 b4 = B_vec[col];
        float4 c4;
        c4.x = a * b4.x;
        c4.y = a * b4.y;
        c4.z = a * b4.z;
        c4.w = a * b4.w;
        C_vec[col] = c4;
    }

    // Process any remaining elements
    int offset = vec_limit * 4;
    for (int col = thread_id; col < (M - offset); col += stride) {
        int index = row * M + offset + col;
        C[index] = a * B[index];
    }
}

// Forward function that wraps our CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create an output tensor with the same type and device as B
    auto C = torch::empty({N, M}, B.options());

    // Launch kernel with one block per row, using 256 threads per block
    const int threads = 256;
    diag_matmul_readonly_kernel<<<N, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using direct read-only memory access");
}
