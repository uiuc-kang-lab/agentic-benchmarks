#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel: uses shared memory caching for A and vectorized accesses for B and C (via float4).
// This kernel computes C[i, j] = A[i] * B[i, j] for a diagonal matrix A and a full matrix B.

__global__ void diag_matmul_kernel_combined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Each thread computes a group of 4 columns (if available) for one row.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // For vectorized access, treat columns in groups of 4
    int vecCol = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_cols = M / 4;  // number of complete groups

    // Shared memory: cache one element of A per row in the block
    extern __shared__ float shared_A[];  // Size: blockDim.y floats
    if (row < N) {
        // One thread per row loads the diagonal element A[row] into shared memory
        if (threadIdx.x == 0) {
            shared_A[threadIdx.y] = A[row];
        }
    }
    __syncthreads();

    // Process the vectorized (float4) portion
    if (row < N && vec_cols > 0 && vecCol < vec_cols) {
        float a_val = shared_A[threadIdx.y];
        // Use vectorized load/store (float4) for coalesced memory access
        const float4* B_vec = reinterpret_cast<const float4*>(B);
        float4* C_vec = reinterpret_cast<float4*>(C);
        int index_vec = row * vec_cols + vecCol;  // row-major ordering in vectorized domain
        float4 b_val = B_vec[index_vec];
        float4 c_val;
        c_val.x = a_val * b_val.x;
        c_val.y = a_val * b_val.y;
        c_val.z = a_val * b_val.z;
        c_val.w = a_val * b_val.w;
        C_vec[index_vec] = c_val;
    }

    // Handle remaining columns if M is not a multiple of 4 (fallback loop)
    if (row < N && threadIdx.x == 0) {
        int rem_start = vec_cols * 4;
        for (int col = rem_start; col < M; col++) {
            C[row * M + col] = shared_A[threadIdx.y] * B[row * M + col];
        }
    }
}

// The forward function wraps the kernel launch for use with PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Configure a 2D grid where each thread computes a vector (group of 4 elements) in a row.
    // blockDim.x is number of vector groups (each of size 4) per block and blockDim.y is rows per block.
    const int block_dim_x = 32;  // threads handling vectorized columns
    const int block_dim_y = 8;   // threads handling rows
    dim3 threads(block_dim_x, block_dim_y);

    // Number of complete float4 groups per row
    int vec_cols = M / 4;
    dim3 blocks((vec_cols + block_dim_x - 1) / block_dim_x, (N + block_dim_y - 1) / block_dim_y);

    // Shared memory: one float per row in the block
    size_t shared_mem_size = block_dim_y * sizeof(float);

    diag_matmul_kernel_combined<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined optimized diagonal matrix multiplication using vectorized memory and shared memory caching");
}
