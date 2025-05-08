#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the tile size; 32 ensures alignment with warp size and memory access coalescing.
const int TILE_SIZE = 32;

// This kernel computes C = A * B^T where A (MxK) and B (NxK) are in row-major order.
// C[i,j] = dot( A[i, :], B[j, :] ) and the global memory accesses are arranged to be coalesced.

__global__ void tiled_matmul_transposed_coalesced_kernel(const float* __restrict__ A,
                                                          const float* __restrict__ B,
                                                          float* __restrict__ C,
                                                          int M, int N, int K) {
    // Each block computes a TILE_SIZE x TILE_SIZE submatrix of C
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Compute the row and column index of the element in C this thread will compute
    int row = blockRow * TILE_SIZE + threadIdx.y; // index in A
    int col = blockCol * TILE_SIZE + threadIdx.x; // corresponds to row index in B since C = A * B^T

    // Accumulator for dot product
    float sum = 0.0f;

    // Allocate shared memory for a tile of A and a tile of B (loaded in transposed form)
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over the tiles of A and B along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Index in A: row is fixed, column is t*TILE_SIZE + threadIdx.x
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && A_col < K) {
            // Global memory is accessed in a coalesced manner because consecutive threads (with threadIdx.x) access consecutive columns
            As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // For B, we need to compute B[j, k] where j = col (since C(i,j) uses B[j, :]).
        // To ensure coalesced access when loading, we load the tile of B in transposed order into shared memory.
        int B_row = col; // B's row (since B is used as if transposed)
        int B_col = t * TILE_SIZE + threadIdx.y;  
        if (B_row < N && B_col < K) {
            // Notice the swap of thread indices (threadIdx.x and threadIdx.y) to store the tile in transposed form.
            Bs[threadIdx.x][threadIdx.y] = B[B_row * K + B_col];
        } else {
            Bs[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles together
        // Access As in row-major and Bs in a way that makes the access contiguous: Bs is transposed in shared memory.
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C in global memory. C is stored in row-major order.
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Forward function exposed to PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_matmul_transposed_coalesced_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failure: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled matrix multiplication with transposed B and coalesced memory accesses (CUDA)");
}
