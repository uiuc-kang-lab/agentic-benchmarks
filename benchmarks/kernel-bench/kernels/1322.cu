#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined optimized kernel that uses shared memory for broadcasting the diagonal vector and
// float4 vectorized loads/stores for coalesced access of matrix B and C.

__global__ void diag_matmul_combined_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Identify the row to process
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N) return;
    
    // Allocate shared memory for one diagonal element per row in the block
    extern __shared__ float shared_A[];  // size: blockDim.y * sizeof(float)
    if (threadIdx.x == 0) {
        shared_A[threadIdx.y] = A[row];
    }
    __syncthreads();
    float a_val = shared_A[threadIdx.y];
    
    // Number of full groups of 4 columns that we can vectorize
    int vec_len = M / 4;
    
    // Process the vectorized portion using float4 loads/stores
    // Each thread will handle multiple vectorized elements along the row
    const float4* B_vec = reinterpret_cast<const float4*>(B + row * M);
    float4* C_vec = reinterpret_cast<float4*>(C + row * M);

    // Compute starting vector index based on block and thread indices
    // GridDim.x * blockDim.x is the total number of threads processing the vectorized portion
    for (int col_vec = threadIdx.x + blockIdx.x * blockDim.x; col_vec < vec_len; col_vec += blockDim.x * gridDim.x) {
        float4 b_val = B_vec[col_vec];
        float4 c_val;
        c_val.x = a_val * b_val.x;
        c_val.y = a_val * b_val.y;
        c_val.z = a_val * b_val.z;
        c_val.w = a_val * b_val.w;
        C_vec[col_vec] = c_val;
    }
    
    // Handle remaining columns that are not a multiple of 4
    int remainder_start = vec_len * 4;
    for (int col = remainder_start + threadIdx.x; col < M; col += blockDim.x) {
        C[row * M + col] = a_val * B[row * M + col];
    }
}

// Forward function that wraps the combined CUDA kernel
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

    // Define block dimensions: blockDim.y covers rows, blockDim.x covers vectorized columns
    const int block_dim_x = 32;  // number of threads for vectorized column processing
    const int block_dim_y = 8;   // number of rows processed per block

    // Calculate grid dimensions. For the x-dimension, if no full vectorizable segment exists (M < 4),
    // ensure at least one block is launched.
    int grid_dim_x = (M / 4) > 0 ? ((M / 4 + block_dim_x - 1) / block_dim_x) : 1;
    int grid_dim_y = (N + block_dim_y - 1) / block_dim_y;
    dim3 blocks(grid_dim_x, grid_dim_y);
    dim3 threads(block_dim_x, block_dim_y);

    // Shared memory allocation: one float per row in the block
    size_t shared_mem_size = block_dim_y * sizeof(float);

    diag_matmul_combined_kernel<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined optimized diagonal matrix multiplication using vectorized loads and shared memory broadcasting");
}
