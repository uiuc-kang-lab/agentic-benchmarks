#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// CUDA kernel for lower triangular matrix multiplication with coalesced memory accesses for global loads/stores
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int N) {
    // Allocate shared memory for tiles of A and B
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    // Compute global row and column indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Out-of-bound threads exit early
    if (row >= N || col >= N) return;

    // For lower triangular matrices, C[row, col] is zero when row < col
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;

    // The k dimension ranges from col to row (inclusive) for a valid multiplication
    // We tile the k dimension. The tile index ranges from t_start to t_end
    int t_start = col / TILE_SIZE;
    int t_end   = row / TILE_SIZE;

    for (int t = t_start; t <= t_end; t++) {
        // Load a tile of A: Each thread loads A[row, t*TILE_SIZE + threadIdx.x]
        int colA = t * TILE_SIZE + threadIdx.x;
        if (colA < N && colA <= row) {
            // Global memory access: row is fixed for the block row and colA varies consecutively
            shA[threadIdx.y][threadIdx.x] = A[row * N + colA];
        } else {
            shA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B: Each thread loads B[t*TILE_SIZE + threadIdx.y, col]
        int rowB = t * TILE_SIZE + threadIdx.y;
        if (rowB < N && rowB >= col) {
            // Global memory access: for a fixed rowB, threads (with different threadIdx.x) read consecutive elements
            shB[threadIdx.y][threadIdx.x] = B[rowB * N + col];
        } else {
            shB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the effective k range within this tile
        int tileStart = t * TILE_SIZE;
        int tileEnd = tileStart + TILE_SIZE;  // exclusive
        int k_begin = (tileStart < col) ? col : tileStart;
        int k_end = (tileEnd > (row + 1)) ? (row + 1) : tileEnd;

        // If the entire tile lies within the valid range, unroll the inner loop
        if ((tileStart >= col) && (tileEnd <= (row + 1))) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
            }
        } else {
            for (int k = k_begin; k < k_end; k++) {
                int local_k = k - tileStart;
                sum += shA[threadIdx.y][local_k] * shB[local_k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Write the result to C with coalesced access (threads in a warp write consecutive elements of a row)
    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Coalesced lower triangular matrix multiplication (CUDA)");
}
