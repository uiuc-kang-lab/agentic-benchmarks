#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define the tile size for block decomposition
#define TILE_SIZE 32

/*
  This kernel performs triangular matrix multiplication C = A * B for lower-triangular matrices A and B.
  It returns a full square matrix C where the upper-triangular elements (row < col) are zero.
  Workloads are distributed evenly across threads and blocks by partitioning the grid based on tile location:
    - Off-diagonal blocks (tile_y > tile_x): All threads in the block compute a valid output element using shared memory tiling.
    - Upper-triangular blocks (tile_y < tile_x): These blocks lie entirely in the zero region and are filled with zeros.
    - Diagonal blocks (tile_x == tile_y): Instead of the natural 2D mapping (which would leave nearly half the threads idle),
      a compact 1D indexing is used to evenly distribute the valid lower-triangular elements (local row >= local col)
      among all threads in the block. Each valid element is computed by looping (with a grid‐stride loop) over the
      compacted indices.
  This design avoids underutilization and bottlenecks due to divergent thread workloads across blocks.
*/

__global__ void even_workload_triangular_mm_kernel(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int N) {
    // Determine the tile indices
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int base_col = tile_x * TILE_SIZE;
    int base_row = tile_y * TILE_SIZE;

    // Case 1: Upper-triangular block (tile_y < tile_x): Fill entire tile with zeros
    if (tile_y < tile_x) {
        int local_index = threadIdx.y * TILE_SIZE + threadIdx.x;
        int total_threads = TILE_SIZE * TILE_SIZE;
        int tile_area = TILE_SIZE * TILE_SIZE;
        for (int i = local_index; i < tile_area; i += total_threads) {
            int r = i / TILE_SIZE;
            int c = i % TILE_SIZE;
            int global_row = base_row + r;
            int global_col = base_col + c;
            if (global_row < N && global_col < N) {
                C[global_row * N + global_col] = 0.0f;
            }
        }
        return;
    }

    // Case 2: Off-diagonal lower-triangular block (tile_y > tile_x): All elements in the block are valid
    if (tile_y > tile_x) {
        int r = base_row + threadIdx.y;
        int c = base_col + threadIdx.x;
        if (r < N && c < N) {
            float sum = 0.0f;
            // Use shared memory tiling for the dot product
            __shared__ float sA[TILE_SIZE][TILE_SIZE];
            __shared__ float sB[TILE_SIZE][TILE_SIZE];

            int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
            for (int m = 0; m < numTiles; m++) {
                int a_col = m * TILE_SIZE + threadIdx.x; // column index in A
                int b_row = m * TILE_SIZE + threadIdx.y;   // row index in B

                // Load tile for A[r, k]: Only load if within bounds and respecting lower-triangular structure (r >= k)
                if (a_col < N && r >= a_col)
                    sA[threadIdx.y][threadIdx.x] = A[r * N + a_col];
                else
                    sA[threadIdx.y][threadIdx.x] = 0.0f;

                // Load tile for B[k, c]: Only load if within bounds and respecting lower-triangular structure (k >= c)
                if (b_row < N && b_row >= c)
                    sB[threadIdx.y][threadIdx.x] = B[b_row * N + c];
                else
                    sB[threadIdx.y][threadIdx.x] = 0.0f;

                __syncthreads();

                // Determine valid k indices in this tile
                int tile_start = m * TILE_SIZE;
                int tile_end = tile_start + TILE_SIZE;
                if (tile_end > N) tile_end = N;
                int k_lower = (c > tile_start) ? c : tile_start;      // global k lower bound
                int k_upper = (r < (tile_end - 1)) ? r : (tile_end - 1); // global k upper bound

                int local_start = k_lower - tile_start;
                int local_end   = k_upper - tile_start + 1;
                for (int k = local_start; k < local_end; k++) {
                    sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
                }
                __syncthreads();
            }
            C[r * N + c] = sum;
        }
        return;
    }

    // Case 3: Diagonal block (tile_x == tile_y): Only lower-triangular elements in this tile are valid.
    // Instead of a natural 2D mapping (which would leave nearly half the threads idle), we compact the valid indices
    // into a 1D range and distribute them evenly among all threads in the block.
    // The number of valid elements in a TILE_SIZE x TILE_SIZE lower-triangular block is:
    int valid_count = (TILE_SIZE * (TILE_SIZE + 1)) / 2;
    int thread_count = TILE_SIZE * TILE_SIZE;
    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;

    // Each thread processes a subset of valid elements using a grid-stride style loop
    for (int t = tid; t < valid_count; t += thread_count) {
        // Map the 1D index t to 2D coordinates within the tile's lower-triangular region
        // Using the quadratic formula: local_row = floor((sqrt(8*t + 1) - 1)/2)
        float f = sqrtf(8.0f * t + 1.0f);
        int local_row = (int)((f - 1.0f) * 0.5f);
        int local_col = t - (local_row * (local_row + 1)) / 2;
        int global_row = base_row + local_row;
        int global_col = base_col + local_col;

        if (global_row < N && global_col < N) {
            float sum = 0.0f;
            // Compute the dot product: sum_{k = global_col}^{global_row} A[global_row, k] * B[k, global_col]
            // For diagonal blocks, the dot product length is at most TILE_SIZE, so shared memory tiling is not used here.
            for (int k = global_col; k <= global_row; k++) {
                sum += A[global_row * N + k] * B[k * N + global_col];
            }
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

    int gridDim = (N + TILE_SIZE - 1) / TILE_SIZE;
    dim3 blocks(gridDim, gridDim);
    dim3 threads(TILE_SIZE, TILE_SIZE);

    even_workload_triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Evenly Distributed Workload Triangular Matrix Multiplication (CUDA)");
}
