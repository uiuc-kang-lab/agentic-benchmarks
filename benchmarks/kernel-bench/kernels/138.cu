#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

// Tiling parameters
#define TILE_SIZE 32
#define SMALL_MATRIX_DIM 128

// Macros for input validations
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel implementing split-K matrix multiplication with minimal atomic operations.
// For each output tile, if the K dimension is split across multiple blocks (split_k > 1),
// each block computes a partial sum and uses atomicAdd to accumulate the final result.
// For split_k == 1, the result is written directly without atomics.

__global__ void splitk_atomic_matmul_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K, int split_k) {
    // Identify tile indices
    int tile_row = blockIdx.y;  // which tile row
    int tile_col = blockIdx.x;  // which tile column
    int split_idx = blockIdx.z; // split index along the K dimension

    int row_start = tile_row * TILE_SIZE;
    int col_start = tile_col * TILE_SIZE;

    // Compute how many elements of K each split should process
    int k_per_split = (K + split_k - 1) / split_k;  // ceiling division
    int k_start = split_idx * k_per_split;
    int k_end = min(k_start + k_per_split, K);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = row_start + ty;
    int col = col_start + tx;

    float sum = 0.0f;

    // Loop over the K dimension in tiles within the subrange assigned to this block
    for (int k_tile = k_start; k_tile < k_end; k_tile += TILE_SIZE) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int a_col = k_tile + tx;
        int b_row = k_tile + ty;

        // Load tile of A into shared memory
        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        // Load tile of B into shared memory
        if (b_row < K && col < N)
            Bs[ty][tx] = B[b_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_SIZE; ++k_inner) {
            sum += As[ty][k_inner] * Bs[k_inner][tx];
        }

        __syncthreads();
    }

    // Write the partial result to global memory
    if (row < M && col < N) {
        if (split_k == 1) {
            // Only one block is responsible for this tile, so write directly
            C[row * N + col] = sum;
        } else {
            // Multiple blocks contribute to the same output element; use atomicAdd to avoid race conditions
            atomicAdd(&C[row * N + col], sum);
        }
    }
}

// Host function that selects the split-K strategy based on the K dimension size.
// When K is large, the K dimension is split among multiple blocks to increase parallelism,
// and atomicAdd is used to accumulate results. For smaller K, no atomics are necessary.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    // A is (M x K), B is (K x N); result C is (M x N)
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions()
                       .dtype(A.dtype())
                       .device(A.device())
                       .requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Determine split_k: use split-K strategy for larger K to boost parallelism.
    int split_k = 1;
    if (K > SMALL_MATRIX_DIM)
        split_k = (K + SMALL_MATRIX_DIM - 1) / SMALL_MATRIX_DIM;  // e.g., if K=256, then split_k=2

    // If split_k > 1, initialize C to zero to safely use atomicAdd
    if (split_k > 1) {
        cudaMemset(d_C, 0, M * N * sizeof(float));
    }

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE,
                  split_k);

    splitk_atomic_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, split_k);
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Split-K atomic matrix multiplication (CUDA) with minimal global atomics");
}
