#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block dimensions for 2D tiling: threads per row and rows per block
#define THREADS_PER_ROW 256
#define ROWS_PER_BLOCK 4

// Kernel for matrix-vector multiplication using a 2D block configuration
// Each block processes ROWS_PER_BLOCK rows concurrently

template <typename scalar_t>
__global__ void matvec_mul_kernel_multirow(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int64_t M,
    const int64_t K) {

    // Compute the row index from 2D block indexing
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    if (row >= M) return;  // Out of bounds, exit

    // Each thread in the row computes a partial sum over columns
    scalar_t partial_sum = 0;
    for (int col = threadIdx.x; col < K; col += THREADS_PER_ROW) {
        // Use __ldg to utilize the read-only data cache
        partial_sum += __ldg(&A[row * K + col]) * __ldg(&B[col]);
    }

    // Use warp-level reduction to reduce partial sums within each warp.
    int lane = threadIdx.x & 31;  // lane index within the warp
    int warpId = threadIdx.x >> 5;  // warp index within the row
    constexpr int numWarps = THREADS_PER_ROW / 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // Allocate shared memory to store warp-level partial sums for each row.
    extern __shared__ scalar_t sdata[];
    if (lane == 0) {
        sdata[threadIdx.y * numWarps + warpId] = partial_sum;
    }
    __syncthreads();

    // The first thread of the first warp in each row accumulates the warp sums.
    if (threadIdx.x == 0) {
        scalar_t sum = 0;
        for (int i = 0; i < numWarps; i++) {
            sum += sdata[threadIdx.y * numWarps + i];
        }
        C[row] = sum;
    }
}

// C++ function wrapping the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure contiguous layout for proper memory access
    A = A.contiguous();
    B = B.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);

    // B should be a flat vector
    auto B_flat = B.view({-1});

    // Allocate output tensor for the result
    auto C = torch::zeros({M}, A.options());

    // Configure 2D block layout
    dim3 block(THREADS_PER_ROW, ROWS_PER_BLOCK);
    // Each block processes ROWS_PER_BLOCK rows, so grid size is computed accordingly
    int grid_x = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    dim3 grid(grid_x);

    // Calculate shared memory size required per block
    size_t shared_mem_size = THREADS_PER_ROW * ROWS_PER_BLOCK * sizeof(float);
    // Launch kernel with AT_DISPATCH_FLOATING_TYPES to handle all floating point types
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_multirow<scalar_t><<<grid, block, shared_mem_size>>>(
            A.data_ptr<scalar_t>(),
            B_flat.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    }));

    return C.view({M, 1});
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}
