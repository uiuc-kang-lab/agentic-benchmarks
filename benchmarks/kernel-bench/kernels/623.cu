#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp and block parameters
#define WARP_SIZE 32
#define BLOCK_ROWS 8  // number of warps (i.e., rows) per block

// CUDA kernel that uses warp-level primitives (shfl) to reduce the dot product
// Each warp computes two output elements (two adjacent columns) for one row in C

template <typename scalar_t>
__global__ void matmul_warp_shuffle_kernel(const scalar_t* __restrict__ A,
                                             const scalar_t* __restrict__ B,
                                             scalar_t* __restrict__ C,
                                             int M, int K, int N) {
    // Each warp computes one row of C, and each warp computes two adjacent columns
    // Block configuration: blockDim.x = WARP_SIZE, blockDim.y = number of warps per block (BLOCK_ROWS)
    // Grid dims: grid.x = ceil(N/2) since each warp computes 2 output columns
    //            grid.y = ceil(M / BLOCK_ROWS)

    int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;  // row index for C
    int col_base = blockIdx.x * 2;                     // starting column index (each warp computes 2 cols)

    // Each thread in a warp holds a partial sum for two output elements
    scalar_t sum0 = 0;
    scalar_t sum1 = 0;

    // Loop over the K dimension in steps of WARP_SIZE
    for (int t = 0; t < K; t += WARP_SIZE) {
        int lane = threadIdx.x; // lane id within the warp (0..31)
        int k = t + lane;      // global index in the K dimension

        // Load element from A for the current row and k
        scalar_t a_val = (row < M && k < K) ? __ldg(&A[row * K + k]) : static_cast<scalar_t>(0);

        // Load elements from B for the current k and for two adjacent columns
        scalar_t b_val0 = (k < K && col_base < N) ? __ldg(&B[k * N + col_base]) : static_cast<scalar_t>(0);
        scalar_t b_val1 = (k < K && (col_base + 1) < N) ? __ldg(&B[k * N + col_base + 1]) : static_cast<scalar_t>(0);

        // Accumulate the product
        sum0 += a_val * b_val0;
        sum1 += a_val * b_val1;
    }

    // Warp-level reduction using __shfl_down_sync for each partial sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
    }

    // Lane 0 writes the final results for this warp
    if (threadIdx.x == 0) {
        if (row < M && col_base < N) {
            C[row * N + col_base] = sum0;
        }
        if (row < M && (col_base + 1) < N) {
            C[row * N + col_base + 1] = sum1;
        }
    }
}

// Host function called by Pybind11
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    // Setup execution configuration
    dim3 block(WARP_SIZE, BLOCK_ROWS);  // 32 x BLOCK_ROWS threads per block
    dim3 grid((N + 1) / 2, (M + BLOCK_ROWS - 1) / BLOCK_ROWS);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_warp_shuffle_kernel", ([&] {
        matmul_warp_shuffle_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));

    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication using warp-level primitives (CUDA)");
}
