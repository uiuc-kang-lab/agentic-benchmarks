#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Unified kernel that selects between vectorized and row-based scalar approaches
__global__ void unified_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M,
    const bool use_vectorized
) {
    if (use_vectorized) {
        // Vectorized branch: works when each row's length M is divisible by 4
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        // Total number of elements in C
        int64_t total = N * M;
        // Each float4 covers 4 consecutive floats
        int64_t vec_total = total / 4;

        // Cast B and C pointers to float4
        const float4* B_vec = reinterpret_cast<const float4*>(B);
        float4* C_vec = reinterpret_cast<float4*>(C);

        for (; idx < vec_total; idx += stride) {
            int base_idx = idx * 4;  // Corresponding starting index in the original array
            int row = base_idx / M;  // Determine the row based on the flat index
            float a_val = A[row];
            
            float4 b_val = B_vec[idx];
            float4 c_val;
            c_val.x = a_val * b_val.x;
            c_val.y = a_val * b_val.y;
            c_val.z = a_val * b_val.z;
            c_val.w = a_val * b_val.w;
            
            C_vec[idx] = c_val;
        }
    } else {
        // Scalar row-based branch using grid-stride loop over rows.
        // Each block will iterate over rows, and threads in the block will collaborate on processing
        // columns within a row for improved memory coalescing.
        for (int row = blockIdx.x; row < N; row += gridDim.x) {
            float a_val = A[row];
            int row_offset = row * M;
            for (int col = threadIdx.x; col < M; col += blockDim.x) {
                int idx = row_offset + col;
                C[idx] = a_val * B[idx];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    const int64_t N = A.size(0);
    const int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Decide which approach to use:
    // Use the vectorized method if M is divisible by 4 and sufficiently large (e.g., M >= 512) 
    // to better harness memory throughput.
    bool use_vectorized = (M % 4 == 0) && (M >= 512);

    if (use_vectorized) {
        const int threads = 256;
        int64_t total = N * M;
        int64_t vec_total = total / 4;
        int blocks = (vec_total + threads - 1) / threads;
        // Clamp grid dimension to hardware limits (max 65535 in x dimension)
        blocks = (blocks > 65535) ? 65535 : blocks;
        unified_diag_matmul_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            N, M, true);
    } else {
        // For the scalar branch, use a grid-stride loop over rows for improved coalescing
        int threads = (M < 256) ? (((M + 31) / 32) * 32) : 256;
        int blocks = (N < 256) ? N : 256;
        unified_diag_matmul_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            N, M, false);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unified diagonal matrix multiplication using vectorized and row-based kernels");
}
