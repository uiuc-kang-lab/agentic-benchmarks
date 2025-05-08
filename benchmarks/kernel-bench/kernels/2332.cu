#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// This kernel computes one output element (C[m][n]) per warp using warp-level reduction.
// To minimize warp divergence, we refactor conditional logic so that the validity check
// for accessing global memory is done once per warp instead of inside the loop.

__global__ void matmul_uniform_warp_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Compute a linear thread index within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;    // Each warp has 32 threads
    int lane = tid % WARP_SIZE;

    // Define tile shape computed per block in terms of warps
    // TILE_M x TILE_N tile per block, where each warp computes one output element
    const int TILE_M = 8;
    const int TILE_N = 4;

    // Determine warp position within the tile
    int warp_row = warp_id / TILE_N;  // Row index for warp in the block tile
    int warp_col = warp_id % TILE_N;  // Column index for warp in the block tile

    // Compute global output indices
    int m = blockIdx.y * TILE_M + warp_row;
    int n = blockIdx.x * TILE_N + warp_col;

    float sum = 0.0f;

    // Check validity once for the entire warp to avoid divergence within the loop
    if (m < M && n < N) {
        // Each thread in the warp processes a subset of the K dimension with stride WARP_SIZE
        for (int k = lane; k < K; k += WARP_SIZE) {
            float a_val = __ldg(&A[m * K + k]);
            float b_val = __ldg(&B[n * K + k]);
            sum += a_val * b_val;
        }

        // Warp-level reduction using __shfl_down_sync
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // The first lane in the warp writes the final result
        if (lane == 0) {
            C[m * N + n] = sum;
        }
    }
}


torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Define tile dimensions
    constexpr int TILE_M = 8;
    constexpr int TILE_N = 4;

    // Each block has 32x32 threads (1024 threads per block), corresponding to 32 warps.
    dim3 block(32, 32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_uniform_warp_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using uniform control flow to minimize warp divergence (CUDA)");
}
