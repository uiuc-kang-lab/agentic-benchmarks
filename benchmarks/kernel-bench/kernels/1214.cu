#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void einsum_kernel_unrolled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= BATCH * I * J * K) return;

    int k = global_idx % K;
    int remainder = global_idx / K;
    int j = remainder % J;
    remainder /= J;
    int i = remainder % I;
    int b = remainder / I;

    float sum = 0.0f;
    
    // Manual loop unrolling for L dimension
    int l = 0;
    #pragma unroll 4
    for(; l + 3 < L; l += 4) {
        int a_offset0 = b * I*J*L + i*J*L + j*L + l;
        int a_offset1 = a_offset0 + 1;
        int a_offset2 = a_offset0 + 2;
        int a_offset3 = a_offset0 + 3;
        
        int b_offset0 = l*K + k;
        int b_offset1 = (l+1)*K + k;
        int b_offset2 = (l+2)*K + k;
        int b_offset3 = (l+3)*K + k;
        
        sum += A[a_offset0] * B[b_offset0] +
               A[a_offset1] * B[b_offset1] +
               A[a_offset2] * B[b_offset2] +
               A[a_offset3] * B[b_offset3];
    }
    
    // Handle remaining elements
    for(; l < L; l++) {
        int a_offset = b * I*J*L + i*J*L + j*L + l;
        int b_offset = l*K + k;
        sum += A[a_offset] * B[b_offset];
    }
    
    C[global_idx] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    int total_elements = BATCH * I * J * K;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    einsum_kernel_unrolled<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with unrolled loops (CUDA)");
}