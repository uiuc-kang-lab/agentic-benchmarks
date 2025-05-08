#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using warp-level primitives for reduction
template <typename scalar_t>
__global__ void matmul_transpose_warp_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    // Each warp computes one element of the output matrix C
    int warpId = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31; // lane index in the warp

    // Total number of elements in C
    int total_elements = M * N;
    if (warpId >= total_elements) return;

    // Map the warp id to a specific output element index
    int row = warpId / N;
    int col = warpId % N;

    scalar_t sum = 0;
    // Each thread in the warp processes a subset of the reduction dimension K
    for (int k = lane; k < K; k += 32) {
        // A is stored as transposed: (K, M) i.e., access A[k * M + row]
        // B is stored as transposed: (N, K) i.e., access B[col * K + k]
        sum += A[k * M + row] * B[col * K + k];
    }

    // Perform warp-level reduction using __shfl_down_sync to sum partial results
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write the final sum to C from the first lane of the warp
    if (lane == 0) {
        C[row * N + col] = sum;
    }
}

// CUDA interface that launches the warp-optimized kernel
torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Get dimensions: A is (K x M) and B is (N x K) as inputs are transposed
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Create output tensor C of dimensions (M x N)
    auto C = torch::empty({M, N}, A.options());

    // Each warp computes one element of C; calculate total warps and blocks
    const int warp_size = 32;
    const int threads_per_block = 128; // 128 threads per block -> 4 warps per block
    int total_warps = M * N;
    int warps_per_block = threads_per_block / warp_size;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_warp_kernel", ([&] {
        matmul_transpose_warp_kernel<scalar_t><<<blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using warp-level primitives (CUDA)");
}
