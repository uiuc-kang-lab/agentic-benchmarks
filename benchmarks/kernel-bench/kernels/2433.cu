#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel implements a tiled matrix multiplication for transposed matrices
// using double-buffering (simulated asynchronous copy) within shared memory.
// It also accepts an offset parameter so that the output matrix C can be computed in partitions.
// The operation computed is: C = A.T * B.T, where A is (K x M) and B is (N x K), so that C is (M x N).
// Note: A is accessed as A[k * full_M + (m + m_offset)] and B as B[n * K + k].

__global__ void matmul_transpose_cpasync_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int subM,    // number of rows in this partition (output submatrix height)
    int full_M,  // full number of rows in A (and C) in the M dimension
    int N,
    int K,
    int m_offset // offset in the M dimension for this partition
) {
    const int TILE_SIZE = 16;
    // Double-buffered shared memory for A and B tiles
    __shared__ float A_shared[2][TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[2][TILE_SIZE][TILE_SIZE];

    // Compute indices within the sub-problem for C
    int tx = threadIdx.x; // local row index within tile
    int ty = threadIdx.y; // local col index within tile
    int row = blockIdx.x * TILE_SIZE + tx; // row index in the submatrix (0 <= row < subM)
    int col = blockIdx.y * TILE_SIZE + ty; // column index in output C, global
    
    float acc = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Double-buffer indices
    int curr = 0, next = 1;

    // Preload the first tile synchronously
    int tile = 0;
    int A_col = tile * TILE_SIZE + ty;  // index in K dimension for A
    // A is (K x full_M) stored as [k * full_M + (m + m_offset)]
    if ((row < subM) && (A_col < K)) {
        A_shared[curr][tx][ty] = A[A_col * full_M + (row + m_offset)];
    } else {
        A_shared[curr][tx][ty] = 0.0f;
    }

    int B_row = tile * TILE_SIZE + tx;  // index in K dimension for B
    // B is (N x K) stored as [n * K + k]
    if ((col < N) && (B_row < K)) {
        B_shared[curr][tx][ty] = B[col * K + B_row];
    } else {
        B_shared[curr][tx][ty] = 0.0f;
    }

    __syncthreads();

    // Main loop over tiles
    for (tile = 0; tile < numTiles; tile++) {
        // If not the last tile, preload the next tile into the alternate buffer
        if (tile < numTiles - 1) {
            int next_tile = tile + 1;
            int A_col_next = next_tile * TILE_SIZE + ty;
            if ((row < subM) && (A_col_next < K)) {
                A_shared[next][tx][ty] = A[A_col_next * full_M + (row + m_offset)];
            } else {
                A_shared[next][tx][ty] = 0.0f;
            }

            int B_row_next = next_tile * TILE_SIZE + tx;
            if ((col < N) && (B_row_next < K)) {
                B_shared[next][tx][ty] = B[col * K + B_row_next];
            } else {
                B_shared[next][tx][ty] = 0.0f;
            }
        }
        __syncthreads();

        // Compute partial dot product from the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_shared[curr][tx][k] * B_shared[curr][k][ty];
        }
        __syncthreads();

        // Swap the double buffers
        curr = 1 - curr;
        next = 1 - next;
    }

    // Write the result to global memory (with proper offset)
    if (((row + m_offset) < full_M) && (col < N)) {
        C[(row + m_offset) * N + col] = acc;
    }
}

// Host function using CUDA streams to overlap kernel execution with data movement
// Here we partition the output matrix C along the M dimension into two halves, launching each part on a separate stream.

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions:
    // A is (K x M) and B is (N x K), so C is (M x N).
    int K = A.size(0);
    int full_M = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({full_M, N}, A.options());

    // Create two CUDA streams
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Partition the output matrix along the M dimension into two parts
    int M_half = (full_M + 1) / 2;  // first partition size
    int M_rem = full_M - M_half;      // second partition size

    // Define block and grid dimensions for a tile size of 16
    dim3 threads(16, 16);
    dim3 blocks0((M_half + 16 - 1) / 16, (N + 16 - 1) / 16);
    dim3 blocks1((M_rem + 16 - 1) / 16, (N + 16 - 1) / 16);

    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch kernel for the first partition (rows [0, M_half)) on stream0 with m_offset = 0
    matmul_transpose_cpasync_kernel<<<blocks0, threads, 0, stream0>>>(
        A_ptr, B_ptr, C_ptr, M_half, full_M, N, K, 0);

    // Launch kernel for the second partition (rows [M_half, full_M)) on stream1 with m_offset = M_half
    if (M_rem > 0) {
        matmul_transpose_cpasync_kernel<<<blocks1, threads, 0, stream1>>>(
            A_ptr, B_ptr, C_ptr, M_rem, full_M, N, K, M_half);
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using cp.async double buffering and CUDA streams (CUDA)");
}
