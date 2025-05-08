#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// This kernel computes one output element (c[m][n]) per warp using warp-level reduction
// Each warp cooperatively computes the dot product for one output element:
//   c[m][n] = sum_{k=0}^{K-1} A[m,k] * B[n,k]
// The output matrix dimensions are M x N, where B is stored in row-major order with shape (N, K).

__global__ void matmul_warp_optimized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Compute a linear thread index within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    int warp_id = tid / WARP_SIZE;    // Each warp has 32 threads
    int lane = tid % WARP_SIZE;         

    // Define tile shape per block in terms of warps
    // Each block computes a tile of output of size TILE_M x TILE_N, where (TILE_M * TILE_N) is the number of warps per block
    // For our configuration, we choose TILE_M = 8 and TILE_N = 4, so there are 32 warps per block (8*4 = 32)
    const int TILE_M = 8;
    const int TILE_N = 4;

    // Determine the row and column position within the block tile for this warp
    int warp_row = warp_id / TILE_N;  // Row index of the warp within the block tile
    int warp_col = warp_id % TILE_N;  // Column index of the warp within the block tile

    // Compute global output indices
    int m = blockIdx.y * TILE_M + warp_row;  // Row index in output matrix
    int n = blockIdx.x * TILE_N + warp_col;    // Column index in output matrix

    float sum = 0.0f;
    // Each thread in the warp processes a subset of the K dimension with a stride of WARP_SIZE
    for (int k = lane; k < K; k += WARP_SIZE) {
        // Load elements from A and B using __ldg for read-only cache optimization
        float a_val = (m < M) ? __ldg(&A[m * K + k]) : 0.0f;
        float b_val = (n < N) ? __ldg(&B[n * K + k]) : 0.0f;
        sum += a_val * b_val;
    }

    // Warp-level reduction using __shfl_down_sync to sum up the partial results from all lanes in the warp.
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // The first lane in the warp writes the final result
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

    // Define the output tile computed by each block in terms of warps
    // We choose TILE_M = 8 (rows) and TILE_N = 4 (columns), so each block computes an 8x4 tile
    constexpr int TILE_M = 8;
    constexpr int TILE_N = 4;

    // Launch configuration:
    // Each block has 32 (WARP_SIZE) threads per warp and 32 warps per block => blockDim = (32, 32) = 1024 threads per block.
    // Grid dimensions cover the output matrix tile by tile.
    dim3 block(32, 32); 
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_warp_optimized_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using warp-level reduction and __ldg (CUDA)");
}
