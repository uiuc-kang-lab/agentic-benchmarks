#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// This kernel computes one output element (C[m][n]) per warp using warp-level reduction
// Each warp cooperatively computes the dot product for one output element:
//   C[m][n] = sum_{k=0}^{K-1} A[m,k] * B[n,k]
// B is stored in row-major order with shape (N, K) to simulate B^T in the multiplication

__global__ void matmul_unroll_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Compute a linear thread index within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    int warp_id = tid / WARP_SIZE;    // Each warp has 32 threads
    int lane = tid % WARP_SIZE;         

    // Define the tile dimensions per block in terms of warps
    // Here, each block computes an 8x4 tile of the output matrix
    const int TILE_M = 8;
    const int TILE_N = 4;

    // Determine the warp's row and column position within the block tile
    int warp_row = warp_id / TILE_N;  // Row index in the tile
    int warp_col = warp_id % TILE_N;  // Column index in the tile

    // Compute global output indices
    int m = blockIdx.y * TILE_M + warp_row;  // Row index in output matrix C
    int n = blockIdx.x * TILE_N + warp_col;    // Column index in output matrix C

    float sum = 0.0f;

    // Unroll the loop over the K dimension to reduce loop overhead
    #pragma unroll
    for (int k = lane; k < K; k += WARP_SIZE) {
        // Use __ldg to load data via the read-only cache
        float a_val = (m < M) ? __ldg(&A[m * K + k]) : 0.0f;
        float b_val = (n < N) ? __ldg(&B[n * K + k]) : 0.0f;
        sum += a_val * b_val;
    }

    // Unroll the warp-level reduction loop
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write the final result for this output element
    if (lane == 0 && m < M && n < N) {
        C[m * N + n] = sum;
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

    // Each block computes an 8x4 tile of the output matrix, with each tile containing 32 warps
    constexpr int TILE_M = 8;
    constexpr int TILE_N = 4;

    // Launch configuration: 32x32 threads per block (1024 threads per block)
    dim3 block(32, 32);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_unroll_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using unrolled loops (CUDA)");
}
