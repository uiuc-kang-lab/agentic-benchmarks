#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 4

__global__ void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    // Calculate global indices
    const int k = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int batch_i_idx = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z;
    
    const int b = batch_i_idx / I;
    const int i = batch_i_idx % I;
    
    // Early exit if out of bounds
    if (k >= K || j >= J || b >= BATCH) return;
    
    // Use register to accumulate sum
    float sum = 0.0f;
    
    // Precompute offsets
    const int a_batch_offset = b * I * J * L + i * J * L + j * L;
    const int c_idx = b * I * J * K + i * J * K + j * K + k;
    
    // Main computation loop with vectorized memory access
    #pragma unroll 4
    for (int l = 0; l < L; l++) {
        sum += A[a_batch_offset + l] * B[l * K + k];
    }
    
    // Store result
    C[c_idx] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    // 3D thread block configuration
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    
    // Calculate grid dimensions
    dim3 blocks(
        (K + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (J + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        ((BATCH * I) + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );
    
    einsum_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication (CUDA)");
}