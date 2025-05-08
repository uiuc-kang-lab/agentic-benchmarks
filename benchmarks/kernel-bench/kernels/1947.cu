#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define configuration parameters
#define WARP_SIZE 32
#define TILE_M 8      // Number of rows per tile
#define TILE_N 4      // Number of columns per tile
#define UNROLL 4      // Unroll factor for inner loop

// Optimized kernel using warp-level reduction and loop unrolling with __ldg to leverage the read-only cache
__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int N) {
    // Each block is organized with blockDim.x = WARP_SIZE and blockDim.y = TILE_M * TILE_N,
    // so that each warp (of 32 threads) computes one output element of C.
    int lane = threadIdx.x;         // Lane index within the warp [0, WARP_SIZE-1]
    int warp_id = threadIdx.y;        // Warp id within the block tile

    // Map the warp id to 2D tile coordinates:
    int warp_row = warp_id / TILE_N;  // Row offset within the block tile
    int warp_col = warp_id % TILE_N;  // Column offset within the block tile

    // Compute global matrix coordinates
    int global_row = blockIdx.y * TILE_M + warp_row;
    int global_col = blockIdx.x * TILE_N + warp_col;

    // Bounds check
    if (global_row >= N || global_col >= N) {
        return;
    }

    // For the triangular matrix multiplication, only the lower triangular part is computed.
    // If global_row < global_col then the output is zero.
    if (global_row < global_col) {
        if (lane == 0) {
            C[global_row * N + global_col] = 0.f;
        }
        return;
    }

    // Each warp computes C[global_row, global_col] = sum_{k=global_col}^{global_row} A[global_row,k]*B[k,global_col]
    float sum = 0.f;

    // Each lane starts at a different offset in the k loop (to split the work among the 32 lanes).
    int k = global_col + lane;

    // Use loop unrolling to accumulate UNROLL iterations per loop, when possible.
    // Make sure that k + (UNROLL-1)*WARP_SIZE is within bounds
    while (k + (UNROLL - 1) * WARP_SIZE <= global_row) {
        sum += __ldg(&A[global_row * N + k])                * __ldg(&B[k * N + global_col]);
        sum += __ldg(&A[global_row * N + k + WARP_SIZE])       * __ldg(&B[(k + WARP_SIZE) * N + global_col]);
        sum += __ldg(&A[global_row * N + k + 2 * WARP_SIZE])   * __ldg(&B[(k + 2 * WARP_SIZE) * N + global_col]);
        sum += __ldg(&A[global_row * N + k + 3 * WARP_SIZE])   * __ldg(&B[(k + 3 * WARP_SIZE) * N + global_col]);
        k += WARP_SIZE * UNROLL;
    }

    // Process any remaining iterations
    while (k <= global_row) {
        sum += __ldg(&A[global_row * N + k]) * __ldg(&B[k * N + global_col]);
        k += WARP_SIZE;
    }

    // Use warp-level shuffle reduction to sum the partial results from all lanes
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Lane 0 writes the final result for this C element
    if (lane == 0) {
        C[global_row * N + global_col] = sum;
    }
}

// C++ interface for PyTorch
extern "C" at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Each block computes a TILE_M x TILE_N tile of the output matrix,
    // with each warp (of 32 threads) computing one element of the tile.
    dim3 block(WARP_SIZE, TILE_M * TILE_N);
    int grid_x = (N + TILE_N - 1) / TILE_N;
    int grid_y = (N + TILE_M - 1) / TILE_M;
    dim3 grid(grid_x, grid_y);

    optimized_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}
