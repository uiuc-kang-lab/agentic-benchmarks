#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    extern __shared__ float shared_B[];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = BATCH * I * J * K;
    if (global_idx >= total_elements) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    float sum = 0.0f;

    // Load B into shared memory
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        shared_B[l * K + k] = B[l * K + k];
    }
    __syncthreads();

    for (int l = 0; l < L; ++l) {
        int a_offset = b * I * J * L + i * J * L + j * L + l;
        sum += A[a_offset] * shared_B[l * K + k];
    }
    C[global_idx] = sum;
}

// Forward function to launch the kernel with shared memory optimization
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
    size_t shared_mem_size = L * K * sizeof(float);

    einsum_kernel_shared<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with shared memory optimization (CUDA)");
}
