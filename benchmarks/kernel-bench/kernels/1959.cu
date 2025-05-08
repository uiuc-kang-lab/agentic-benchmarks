#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel uses double-buffered shared memory to load tiles from A and B
// and minimizes __syncthreads() usage by requiring only one synchronization per tile load.
// Each tile's data is loaded into one of two shared memory buffers, and the computation
// is performed immediately after a single sync, thus reducing overhead.

__global__ void minimized_syncs_db_triangular_mm(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Out-of-bound threads return early
    if (row >= N || col >= N) return;
    // For upper-triangular elements, enforce zero
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Double-buffered shared memory: two buffers to alternate tile loads
    __shared__ float sA[2][TILE_SIZE][TILE_SIZE];
    __shared__ float sB[2][TILE_SIZE][TILE_SIZE];

    // Process first tile (m = 0) using buffer 0
    {
        int m = 0;
        int tile_start = m * TILE_SIZE;
        int tile_end = tile_start + TILE_SIZE;
        if (tile_end > N) tile_end = N;
        int buf = 0;
        
        int idxA = tile_start + threadIdx.x;
        sA[buf][threadIdx.y][threadIdx.x] = (idxA < N && row >= idxA) ? A[row * N + idxA] : 0.0f;

        int idxB = tile_start + threadIdx.y;
        sB[buf][threadIdx.y][threadIdx.x] = (idxB < N && idxB >= col) ? B[idxB * N + col] : 0.0f;

        __syncthreads();

        // Determine local k range for valid lower-triangular products
        int local_start = (col > tile_start) ? (col - tile_start) : 0;
        int local_end = (row < tile_end) ? (row - tile_start) : (tile_end - tile_start - 1);
        for (int k = local_start; k <= local_end; k++) {
            sum += sA[buf][threadIdx.y][k] * sB[buf][k][threadIdx.x];
        }
    }

    // Process remaining tiles using double buffering to minimize synchronizations
    for (int m = 1; m < numTiles; m++) {
        int buf = m & 1;  // Alternate buffer index: 0 or 1
        int tile_start = m * TILE_SIZE;
        int tile_end = tile_start + TILE_SIZE;
        if (tile_end > N) tile_end = N;

        int idxA = tile_start + threadIdx.x;
        sA[buf][threadIdx.y][threadIdx.x] = (idxA < N && row >= idxA) ? A[row * N + idxA] : 0.0f;

        int idxB = tile_start + threadIdx.y;
        sB[buf][threadIdx.y][threadIdx.x] = (idxB < N && idxB >= col) ? B[idxB * N + col] : 0.0f;

        __syncthreads();

        int local_start = (col > tile_start) ? (col - tile_start) : 0;
        int local_end = (row < tile_end) ? (row - tile_start) : (tile_end - tile_start - 1);
        for (int k = local_start; k <= local_end; k++) {
            sum += sA[buf][threadIdx.y][k] * sB[buf][k][threadIdx.x];
        }
    }

    C[row * N + col] = sum;
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    minimized_syncs_db_triangular_mm<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Double Buffered Lower Triangular Matrix Multiplication with Minimized Syncthreads");
}
