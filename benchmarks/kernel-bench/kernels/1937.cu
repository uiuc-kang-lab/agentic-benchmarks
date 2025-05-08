#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_M 8      // Number of output rows computed per block tile
#define TILE_N 4      // Number of output columns computed per block tile

// Each warp computes one element C[i,j] for a lower triangular matrix multiplication
// C[i,j] = sum_{k=j}^{i} A[i,k] * B[k,j] if i >= j, else 0.
// The warp partitions the summation among its lanes and then uses warp-level shuffle reduction.
__global__ void warp_reduction_triangular_mm_kernel(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int N) {
    // Each block computes a tile of size TILE_M x TILE_N (output elements)
    // Each output element is computed cooperatively by one warp (WARP_SIZE threads).
    // Block configuration: blockDim.x = WARP_SIZE, blockDim.y = TILE_M * TILE_N
    
    int lane = threadIdx.x;            // Lane id within a warp [0,31]
    int warp_id = threadIdx.y;           // Warp id within block; each warp computes one output element

    // Map warp_id to tile coordinates within the block tile
    int warp_row = warp_id / TILE_N;     // row index offset within the block tile
    int warp_col = warp_id % TILE_N;     // column index offset within the block tile

    int global_row = blockIdx.y * TILE_M + warp_row;
    int global_col = blockIdx.x * TILE_N + warp_col;

    // Check bounds. Also, for positions in the upper triangular region, we set C to 0.
    if (global_row < N && global_col < N) {
        if (global_row < global_col) {
            if (lane == 0) {
                C[global_row * N + global_col] = 0.f;
            }
            return;
        }

        float sum = 0.f;
        // The reduction range for k goes from global_col to global_row (inclusive)
        // Each lane processes a subset of the indices in strides of WARP_SIZE
        for (int k = global_col + lane; k <= global_row; k += WARP_SIZE) {
            float a_val = A[global_row * N + k];
            float b_val = B[k * N + global_col];
            sum += a_val * b_val;
        }

        // Use warp-level shuffle reduction to sum partial results from all lanes
        unsigned int mask = 0xffffffff;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        // Lane 0 writes the result
        if (lane == 0) {
            C[global_row * N + global_col] = sum;
        }
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Each block computes a tile of output of dimensions TILE_M x TILE_N
    // Each output element is computed by one warp (WARP_SIZE threads).
    // Therefore, block dimensions: x-dim = WARP_SIZE, y-dim = TILE_M*TILE_N
    dim3 block(WARP_SIZE, TILE_M * TILE_N);
    int grid_x = (N + TILE_N - 1) / TILE_N; // one block tile covers TILE_N columns
    int grid_y = (N + TILE_M - 1) / TILE_M; // one block tile covers TILE_M rows
    dim3 grid(grid_x, grid_y);

    warp_reduction_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level reduction based lower triangular matrix multiplication (CUDA)");
}
