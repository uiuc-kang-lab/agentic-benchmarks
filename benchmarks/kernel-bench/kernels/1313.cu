#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Optimized CUDA kernel that multiplies a diagonal matrix A with matrix B.
// Each output element is computed as C[i, j] = A[i] * B[i, j].
// Optimization strategies:
// 1. Instead of each thread loading A[i] from global memory, one thread per row in the block loads A[i] into shared memory.
// 2. Within each warp, __shfl_sync is used to efficiently broadcast the loaded A value to all threads, reducing redundant global memory accesses.

__global__ void diag_matmul_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Calculate the row and column indices of the element this thread will compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        // Allocate shared memory for the diagonal elements corresponding to each row in the block
        extern __shared__ float shared_A[]; // size: blockDim.y * sizeof(float)
        // Have one thread per row (e.g., threadIdx.x == 0) load A[row] into shared memory
        if (threadIdx.x == 0) {
            shared_A[threadIdx.y] = A[row];
        }
        __syncthreads();

        // Use warp-level primitives to broadcast the diagonal value within each warp
        unsigned int mask = __activemask();
        float a_val = shared_A[threadIdx.y];
        // Broadcast the value from lane 0 of each warp (the value is identical for the entire row)
        a_val = __shfl_sync(mask, a_val, 0);

        // Load the corresponding element from matrix B
        float b_val = B[row * M + col];

        // Compute the product
        float c_val = a_val * b_val;

        // Write the result back to global memory
        C[row * M + col] = c_val;
    }
}

// Forward function that wraps our optimized CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create an output tensor with the same device and type as B
    auto C = torch::empty({N, M}, B.options());

    // Define block dimensions. Using a 2D block, e.g., 32 threads in x and 8 threads in y.
    const int block_dim_x = 32;
    const int block_dim_y = 8;
    dim3 threads(block_dim_x, block_dim_y);
    dim3 blocks((M + block_dim_x - 1) / block_dim_x, (N + block_dim_y - 1) / block_dim_y);

    // Allocate shared memory: one float per row in the block (blockDim.y floats per block)
    size_t shared_mem_size = block_dim_y * sizeof(float);

    // Launch the optimized kernel
    diag_matmul_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Create the PyTorch extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matrix multiplication of A and B on the GPU");
}
