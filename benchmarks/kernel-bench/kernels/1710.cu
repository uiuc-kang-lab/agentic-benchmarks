#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define NUM_STREAMS 4

// Kernel to compute C = tril(A * B) for lower triangular matrices using tiling and stream-based pipelining.
// The kernel processes a chunk of rows starting at 'start_row'.
__global__ void triangular_mm_kernel_stream(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N,
                                             int start_row) {
    // Compute global row and column indices for C
    int row = start_row + blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    // For upper-triangular region, output zero
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int tile_start = t * TILE_SIZE;
        int tile_end = tile_start + TILE_SIZE;
        if (tile_end > N) tile_end = N;

        // If the tile's k-range starts beyond the current row, no further contributions
        if (tile_start > row) break;

        // Load a tile of A: row index is fixed, column varies
        int a_col = tile_start + threadIdx.x;
        if (a_col < N) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + a_col]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.f;
        }

        // Load a tile of B: column index is fixed, row varies
        int b_row = tile_start + threadIdx.y;
        if (b_row < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[b_row * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.f;
        }
        __syncthreads();

        // Determine the valid range for summation in this tile
        int k_start = (tile_start > col) ? tile_start : col;
        int k_end   = (tile_end < (row + 1)) ? tile_end : (row + 1);

        for (int k = k_start; k < k_end; k++) {
            int k_local = k - tile_start;
            sum += As[threadIdx.y][k_local] * Bs[k_local][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
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

    // Divide the work along the row dimension across multiple streams
    int chunk = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
    }

    dim3 block(TILE_SIZE, TILE_SIZE);
    // The grid for each stream is computed based on the chunk of rows handled by that stream
    for (int s = 0; s < NUM_STREAMS; s++) {
        int start_row = s * chunk;
        int rows = ((start_row + chunk) > N) ? (N - start_row) : chunk;
        if (rows <= 0) continue;
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

        triangular_mm_kernel_stream<<<grid, block, 0, streams[s]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            start_row
        );
    }

    // Synchronize all streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Streamed and Pipelined Triangular Matrix Multiplication (CUDA)");
}
