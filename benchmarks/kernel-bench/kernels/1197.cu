#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel_coalesced(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    int batch_idx = blockIdx.z;
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= J) return;

    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        int a_offset = batch_idx * I * J * L + i * J * L + j * L + l;
        int b_offset = l * K + threadIdx.x;
        sum += A[a_offset] * B[b_offset];
    }

    int c_offset = batch_idx * I * J * K + i * J * K + j * K + threadIdx.x;
    C[c_offset] = sum;
}

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

    dim3 threads(K, 1, 1);
    dim3 blocks((J + threads.x - 1) / threads.x, I, BATCH);

    einsum_kernel_coalesced<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with memory coalescing (CUDA)");
}
