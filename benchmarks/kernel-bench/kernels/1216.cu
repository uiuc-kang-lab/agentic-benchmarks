#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized kernel using a grid-stride loop to ensure uniform control flow across warps
__global__ void einsum_kernel_gridstride(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    int total = BATCH * I * J * K;
    int stride = blockDim.x * gridDim.x;
    
    // Use grid-stride loop to cover all elements without early exit divergence
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int global_idx = idx;
        
        // Decode the 4D tensor indices from global index
        int k = global_idx % K;
        int temp = global_idx / K;
        int j = temp % J;
        temp /= J;
        int i = temp % I;
        int b = temp / I;
        
        float sum = 0.0f;
        // #pragma unroll may help if L is statically known or small
        #pragma unroll
        for (int l = 0; l < L; l++) {
            int a_index = b * I * J * L + i * J * L + j * L + l;
            int b_index = l * K + k;
            sum += A[a_index] * B[b_index];
        }
        
        C[global_idx] = sum;
    }
}

// Forward function to configure and launch the kernel

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
    int total = BATCH * I * J * K;
    
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    einsum_kernel_gridstride<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized grid-stride loop 4D tensor-matrix multiplication (CUDA)");
}
