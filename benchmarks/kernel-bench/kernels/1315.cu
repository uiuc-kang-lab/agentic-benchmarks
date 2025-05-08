#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * M;
    const int vec_total = total / 4;
    
    // Vector types for coalesced memory access
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    float4* C_vec = reinterpret_cast<float4*>(C);
    
    // Process 4 elements at a time using float4
    for (int idx = tid; idx < vec_total; idx += stride) {
        const int base_idx = idx * 4;
        const int row = base_idx / M;
        const float a_val = A[row];
        
        float4 b_val = B_vec[idx];
        float4 c_val;
        c_val.x = a_val * b_val.x;
        c_val.y = a_val * b_val.y;
        c_val.z = a_val * b_val.z;
        c_val.w = a_val * b_val.w;
        
        C_vec[idx] = c_val;
    }
    
    // Handle remaining elements
    const int vec_processed = vec_total * 4;
    for (int idx = vec_processed + tid; idx < total; idx += stride) {
        const int row = idx / M;
        C[idx] = A[row] * B[idx];
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
    const int blocks = min(65535, (int)((N * M + threads * 4 - 1) / (threads * 4)));

    diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized diagonal matrix multiplication");
}