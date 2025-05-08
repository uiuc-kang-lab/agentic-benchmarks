#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// This kernel implements split-K matrix multiplication with transposed B.
// The K dimension is partitioned across the grid's z-dimension. Each block computes
// a partial product for a tile of the output matrix. After computing the partial sum,
// each block accumulates its result into the global C matrix with a single atomicAdd per element,
// minimizing contention in global memory while ensuring correctness.

__global__ void splitk_matmul_transposed_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int M, int N, int K) {
    // Determine tile coordinates in the output matrix
    int tile_row = blockIdx.y * TILE_SIZE;
    int tile_col = blockIdx.x * TILE_SIZE;
    // Partition the K dimension across the grid's z-dimension
    int k_offset = blockIdx.z * TILE_SIZE;

    // Thread indices within the tile
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = tile_row + ty;
    int col = tile_col + tx;

    float sum = 0.0f;

    // Loop over the K tile assigned to this block
    for (int t = 0; t < TILE_SIZE; t++) {
        int k = k_offset + t;
        float a_val = 0.0f;
        float b_val = 0.0f;

        if (row < M && k < K)
            a_val = A[row * K + k];
        if (col < N && k < K)
            // Note: B is used as if transposed, so C[i][j] = dot(A[i], B[j])
            b_val = B[col * K + k];

        sum += a_val * b_val;
    }

    // Use atomicAdd to accumulate the partial sum from this block into the global result
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

// Forward function called from PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "The inner dimensions of A and B must match");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    // Allocate the output tensor and initialize it to zero for proper atomic accumulation
    auto C = torch::zeros({M, N}, A.options());

    // Set up grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              (K + TILE_SIZE - 1) / TILE_SIZE);

    splitk_matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Split-K matrix multiplication with transposed B using minimal atomic operations (CUDA)");
}
