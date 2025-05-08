#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel for when M is divisible by 4: use float4 for coalesced memory accesses
__global__ void coalesced_diag_matmul_vec_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t M
) {
    int row = blockIdx.x; // each block processes one row
    float a_val = A[row];

    // Number of float4 elements per row
    int vec_cols = M / 4;

    // Cast row pointers to float4 pointers
    const float4* B_row = reinterpret_cast<const float4*>(B + row * M);
    float4* C_row = reinterpret_cast<float4*>(C + row * M);

    // Each thread processes several consecutive float4 elements
    for (int v = threadIdx.x; v < vec_cols; v += blockDim.x) {
        float4 b_val = B_row[v];
        float4 c_val;
        c_val.x = a_val * b_val.x;
        c_val.y = a_val * b_val.y;
        c_val.z = a_val * b_val.z;
        c_val.w = a_val * b_val.w;
        C_row[v] = c_val;
    }
}

// Fallback kernel for when M is not divisible by 4: process element-wise
__global__ void coalesced_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t M
) {
    int row = blockIdx.x; // each block processes one row
    float a_val = A[row];
    int offset = row * M;
    
    // Each thread processes elements in the row with a fixed stride
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        C[offset + j] = a_val * B[offset + j];
    }
}

// Forward function that dispatches the appropriate kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch between A and B");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    // If M is divisible by 4 and large enough, use the vectorized kernel
    if (M >= 4 && (M % 4 == 0)) {
        // Use one block per row. Choose thread count based on number of float4 elements
        int threads = (M / 4) < 256 ? (int)(M / 4) : 256;
        dim3 grid(N);
        coalesced_diag_matmul_vec_kernel<<<grid, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M
        );
    } else {
        // Fallback kernel: one block per row, processing element-wise
        int threads = M < 256 ? M : 256;
        dim3 grid(N);
        coalesced_diag_matmul_kernel<<<grid, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M
        );
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with coalesced memory accesses");
}
