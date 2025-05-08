#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for diagonal elements (64KB = 16384 float elements)
#define MAX_DIAG_SIZE 16384
__constant__ float d_diag[MAX_DIAG_SIZE];

__global__ void diag_matmul_const_kernel(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    // Load diagonal element from constant memory
    const float a_val = d_diag[row];
    
    // Process four elements at a time using vectorized loads
    int col = threadIdx.x;
    const int stride = blockDim.x;
    const float4* B_vec = reinterpret_cast<const float4*>(B + row * M);
    float4* C_vec = reinterpret_cast<float4*>(C + row * M);
    
    // Handle vectorized loads first
    const int vec_limit = M / 4;
    while (col < vec_limit) {
        float4 b4 = B_vec[col];
        float4 c4;
        c4.x = a_val * b4.x;
        c4.y = a_val * b4.y;
        c4.z = a_val * b4.z;
        c4.w = a_val * b4.w;
        C_vec[col] = c4;
        col += stride;
    }
    
    // Handle remaining elements
    col = threadIdx.x + (vec_limit * 4);
    while (col < M) {
        C[row * M + col] = a_val * B[row * M + col];
        col += stride;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");
    TORCH_CHECK(A.size(0) <= MAX_DIAG_SIZE,
                "Diagonal matrix too large for constant memory");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Copy diagonal elements to constant memory
    cudaMemcpyToSymbol(d_diag, A.data_ptr<float>(), N * sizeof(float));

    // Create output tensor
    auto C = torch::empty({N, M}, B.options());

    // Launch kernel
    const int threads = 512;
    diag_matmul_const_kernel<<<N, threads>>>(
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using constant memory");
}