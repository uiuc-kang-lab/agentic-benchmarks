#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel computes C = A * B for lower-triangular matrices A and B, where only the lower-triangular
// part of the result is computed (i.e. for row < col, C is set to 0). It uses tiled multiplication with shared
// memory to ensure coalesced global memory accesses. Each tile load is aligned so that threads in a warp
// read consecutive memory locations.

__global__ void triangular_mm_coalesced_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    // Compute global row and column indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // If outside matrix bounds, exit
    if (row >= N || col >= N) return;

    // For upper-triangular region, the multiplication is not needed
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float result = 0.0f;

    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Number of tiles along the k dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Compute global indices for the tile loads
        int A_col = t * TILE_SIZE + threadIdx.x;  // A[row, A_col]
        int B_row = t * TILE_SIZE + threadIdx.y;    // B[B_row, col]

        // Load tile of A: Each thread in a row loads a consecutive element
        if (A_col < N)
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + A_col]);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B: Threads accessing the same row load consecutive elements
        if (B_row < N)
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[B_row * N + col]);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // The tile corresponds to the k indices [t*TILE_SIZE, t*TILE_SIZE + TILE_SIZE).
        // For triangular matrix multiplication, only the range k in [col, row] are valid.
        int tile_k_start = t * TILE_SIZE;
        int tile_k_end = tile_k_start + TILE_SIZE;
        
        // Clamp the valid k range to [col, row+1] (since row is inclusive)
        int valid_start = (col > tile_k_start) ? col : tile_k_start;
        int valid_end = (row + 1 < tile_k_end) ? (row + 1) : tile_k_end;

        // Accumulate partial result for valid k indices
        for (int k = valid_start; k < valid_end; k++) {
            int k_local = k - tile_k_start;  // index within the shared memory tile
            result += As[threadIdx.y][k_local] * Bs[k_local][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed result to global memory. Global stores are coalesced as threads in the same row
    // write to consecutive addresses.
    C[row * N + col] = result;
}

// PyTorch interface function
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_coalesced_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Coalesced Tiled Triangular Matrix Multiplication (CUDA)");
}
