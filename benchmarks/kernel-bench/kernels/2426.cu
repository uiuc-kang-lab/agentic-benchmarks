#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each block computes a partial sum for one output element for a subrange of K.
// Grid is 3D: x and y for output tile indices, z for partitioning the K dimension.
// When gridDim.z > 1, multiple blocks contribute to the same output element, so atomicAdd is used once per block per element.
// If gridDim.z == 1, then each output element is computed in a single block and can be written directly, avoiding atomic overhead.

template <typename scalar_t>
__global__ void matmul_transpose_atomic_kernel(
    const scalar_t* __restrict__ A,  // A is (K x M), with A[k, m] stored as A[k * M + m]
    const scalar_t* __restrict__ B,  // B is (N x K), with B[n, k] stored as B[n * K + k]
    scalar_t* __restrict__ C,        // C is (M x N), with C[m, n] stored as C[m * N + n]
    const int M,                     // number of rows in C (and A's 2nd dimension)
    const int N,                     // number of columns in C (and B's 1st dimension)
    const int K) {                   // summation dimension

    const int TILE_SIZE = 16;  // for output tile dimensions
    const int TILE_K = 16;     // partition size along K dimension

    // Compute output indices (m for row, n for column): Each thread computes one output element.
    int m = blockIdx.y * TILE_SIZE + threadIdx.y;
    int n = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (m >= M || n >= N) return;

    // Determine the subrange of K to process in this block
    int kStart = blockIdx.z * TILE_K;
    int kEnd = (kStart + TILE_K > K) ? K : (kStart + TILE_K);

    scalar_t partial = 0;

    for (int k = kStart; k < kEnd; k++) {
        // A is stored transposed: A[k, m] = A[k * M + m]
        // B is stored transposed: B[n, k] = B[n * K + k]
        partial += A[k * M + m] * B[n * K + k];
    }

    // Use atomicAdd only if multiple blocks (gridDim.z > 1) contribute to the same C element
    if (gridDim.z == 1) {
        C[m * N + n] = partial;
    } else {
        atomicAdd(&C[m * N + n], partial);
    }
}

// CUDA interface exposed to PyTorch
// We partition the K summation dimension across blocks in the z-dimension.
// If more than one block processes a given (m, n), then C must be initialized to zero.

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions: A is (K x M), B is (N x K) so that C is (M x N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    const int TILE_SIZE = 16;
    const int TILE_K = 16;

    // Compute grid dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        (K + TILE_K - 1) / TILE_K
    );

    // If the K dimension is partitioned among multiple blocks, initialize C to zero to enable atomic accumulation.
    torch::Tensor C;
    if (blocks.z > 1) {
        C = torch::zeros({M, N}, A.options());
    } else {
        C = torch::empty({M, N}, A.options());
    }

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_atomic_kernel", ([&] {
        matmul_transpose_atomic_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using minimal atomic operations (CUDA)");
}
