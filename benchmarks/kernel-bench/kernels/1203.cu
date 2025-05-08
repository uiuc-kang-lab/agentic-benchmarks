#include <torch/extension.h>
#include <cuda_runtime.h>

// Combined optimized kernel: uses __ldg() for read-only memory accesses and loop unrolling for improved throughput
__global__ void einsum_kernel_combined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = BATCH * I * J * K;
    if (global_idx >= total_elements) return;

    // Compute indices: global_idx indexes into a 4D tensor of shape (BATCH, I, J, K)
    int k = global_idx % K;
    int tmp = global_idx / K;
    int j = tmp % J;
    tmp /= J;
    int i = tmp % I;
    int b = tmp / I;

    float sum = 0.0f;

    // Loop unrolling factor of 4 for the reduction on dimension L
    int l = 0;
    int L4 = L & ~3;  // largest multiple of 4 less than or equal to L
    for (; l < L4; l += 4) {
        // Compute offsets for tensor A and matrix B
        int a_offset = b * I * J * L + i * J * L + j * L + l;
        int b_offset = l * K + k;

        // Unrolled computations using __ldg() for read-only cache loads
        sum += __ldg(&A[a_offset])     * __ldg(&B[b_offset])
             + __ldg(&A[a_offset + 1]) * __ldg(&B[b_offset + K])
             + __ldg(&A[a_offset + 2]) * __ldg(&B[b_offset + 2*K])
             + __ldg(&A[a_offset + 3]) * __ldg(&B[b_offset + 3*K]);
    }

    // Process remaining elements if L is not a multiple of 4
    for (; l < L; ++l) {
        int a_offset = b * I * J * L + i * J * L + j * L + l;
        int b_offset = l * K + k;
        sum += __ldg(&A[a_offset]) * __ldg(&B[b_offset]);
    }

    C[global_idx] = sum;
}

// Host function to launch the kernel

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    int total_elements = BATCH * I * J * K;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    einsum_kernel_combined<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined optimized 4D tensor-matrix multiplication (CUDA)");
}
