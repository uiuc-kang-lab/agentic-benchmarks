#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized kernel using 3D grid and block indexing
__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Map grid dimensions to our problem domain:
    // - blockIdx.z indexes the batch dimension (BATCH)
    // - blockIdx.y and threadIdx.y index the I dimension
    // - blockIdx.x and threadIdx.x index the J dimension
    // - threadIdx.z (with loop) covers the K dimension
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop over k dimension if K is larger than blockDim.z
    for (int k = threadIdx.z; k < K; k += blockDim.z) {
        if (i < I && j < J && k < K) {
            float sum = 0.0f;
            // Sum over dimension L
            for (int l = 0; l < L; ++l) {
                int a_index = b * (I * J * L) + i * (J * L) + j * L + l;
                int b_index = l * K + k;
                sum += A[a_index] * B[b_index];
            }
            int c_index = b * (I * J * K) + i * (J * K) + j * K + k;
            C[c_index] = sum;
        }
    }
}

// Forward function to launch the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in L");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Set up 3D block and grid dimensions
    // Using block dimensions: 16 threads for j, 16 threads for i, and 4 threads for k (total 1024 if fully occupied)
    dim3 block(16, 16, 4);
    // Grid dimensions: divide J and I among blocks; batch dimension is mapped to grid z
    dim3 grid((J + block.x - 1) / block.x, (I + block.y - 1) / block.y, BATCH);

    einsum_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 4D tensor-matrix multiplication using 3D thread/block indexing (CUDA)");
}
