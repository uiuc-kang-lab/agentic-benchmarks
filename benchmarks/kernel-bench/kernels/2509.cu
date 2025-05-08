#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to load an element from matrix A using read-only cache
template <typename scalar_t>
__device__ inline scalar_t load_a(const scalar_t* __restrict__ A, int k, int row, int M) {
    return __ldg(&A[k * M + row]);
}

// Device function to load an element from matrix B using read-only cache
template <typename scalar_t>
__device__ inline scalar_t load_b(const scalar_t* __restrict__ B, int col, int k, int K) {
    return __ldg(&B[col * K + k]);
}

// Modular device function to compute the dot product for the transposed matrices
// A is stored as (K, M) and B as (N, K), so we access A[k * M + row] and B[col * K + k]
template <typename scalar_t>
__device__ inline scalar_t compute_transposed_dot(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    int row,
    int col,
    int M,
    int K) {

    scalar_t sum = 0;
    #pragma unroll
    int k = 0;
    for (; k <= K - 4; k += 4) {
        sum += load_a<scalar_t>(A, k, row, M) * load_b<scalar_t>(B, col, k, K);
        sum += load_a<scalar_t>(A, k+1, row, M) * load_b<scalar_t>(B, col, k+1, K);
        sum += load_a<scalar_t>(A, k+2, row, M) * load_b<scalar_t>(B, col, k+2, K);
        sum += load_a<scalar_t>(A, k+3, row, M) * load_b<scalar_t>(B, col, k+3, K);
    }
    for (; k < K; ++k) {
        sum += load_a<scalar_t>(A, k, row, M) * load_b<scalar_t>(B, col, k, K);
    }
    return sum;
}

// CUDA kernel that uses the modular device function to compute each element of C
// C = A.T * B.T, where A is (K, M) and B is (N, K)
template <typename scalar_t>
__global__ void matmul_transposed_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, // Number of rows in output matrix C
    int N, // Number of columns in output matrix C
    int K  // Shared inner dimension
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        C[row * N + col] = compute_transposed_dot<scalar_t>(A, B, row, col, M, K);
    }
}

// Host function to launch the CUDA kernel
// A is of shape (K, M) and B is of shape (N, K) as they are transposed
torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Set block and grid sizes
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transposed_kernel", ([&] {
        matmul_transposed_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transposed_cuda, "Modular Matrix multiplication with transposed matrices forward (CUDA)");
}
