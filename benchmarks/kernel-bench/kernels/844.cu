#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Define block and tile dimensions
#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)

// This kernel uses asynchronous copies (cp.async) with double buffering to overlap
// global memory loads into shared memory with computation. Each thread block processes
// a TILE_DIM x TILE_DIM tile of the output matrix, and each thread computes a 2x2 submatrix.

__global__ void async_pipeline_matmul_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int M, int K, int N) {
    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // range [0, BLOCK_SIZE)
    int ty = threadIdx.y; // range [0, BLOCK_SIZE)

    // Each thread computes a 2x2 submatrix of C
    int row0 = by * TILE_DIM + ty;          // first row
    int col0 = bx * TILE_DIM + tx;          // first column
    int row1 = row0 + BLOCK_SIZE;           // second row
    int col1 = col0 + BLOCK_SIZE;           // second column

    float Cvalue00 = 0.0f, Cvalue01 = 0.0f, Cvalue10 = 0.0f, Cvalue11 = 0.0f;

    // Declare double-buffered shared memory for tiles from A and B
    __shared__ float A_shared[2][TILE_DIM][TILE_DIM];
    __shared__ float B_shared[2][TILE_DIM][TILE_DIM];

    const int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    int curr_buf = 0;

    // Prefetch the first tile (tile 0) into buffer 0 using cp.async
    {
      int tile = 0;
      // Load tile for A: each thread loads a 2x2 block
      #pragma unroll
      for (int i = 0; i < 2; i++) {
          int global_row = by * TILE_DIM + ty + i * BLOCK_SIZE;
          #pragma unroll
          for (int j = 0; j < 2; j++) {
              int global_col = tile * TILE_DIM + tx + j * BLOCK_SIZE;
              int shared_row = ty + i * BLOCK_SIZE;
              int shared_col = tx + j * BLOCK_SIZE;
              float *dst = &A_shared[curr_buf][shared_row][shared_col];
              const float *src = A + global_row * K + global_col;
              if (global_row < M && global_col < K) {
                  asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" 
                                : 
                                : "r"(dst), "l"(src), "n"(4));
              } else {
                  *dst = 0.0f;
              }
          }
      }
      // Load tile for B
      #pragma unroll
      for (int i = 0; i < 2; i++) {
          int global_row = tile * TILE_DIM + ty + i * BLOCK_SIZE;
          #pragma unroll
          for (int j = 0; j < 2; j++) {
              int global_col = bx * TILE_DIM + tx + j * BLOCK_SIZE;
              int shared_row = ty + i * BLOCK_SIZE;
              int shared_col = tx + j * BLOCK_SIZE;
              float *dst = &B_shared[curr_buf][shared_row][shared_col];
              const float *src = B + global_row * N + global_col;
              if (global_row < K && global_col < N) {
                  asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" 
                                : 
                                : "r"(dst), "l"(src), "n"(4));
              } else {
                  *dst = 0.0f;
              }
          }
      }
      // Commit and wait for the asynchronous copies to complete
      asm volatile ("cp.async.commit_group;");
      asm volatile ("cp.async.wait_group 0;");
      __syncthreads();
    }

    // Main loop: process each tile and prefetch the next tile asynchronously
    for (int tile = 0; tile < numTiles; tile++) {
        int next_buf = curr_buf ^ 1;
        // Prefetch next tile if available
        if (tile < numTiles - 1) {
            int next_tile = tile + 1;
            // Load next tile for A into buffer next_buf
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int global_row = by * TILE_DIM + ty + i * BLOCK_SIZE;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int global_col = next_tile * TILE_DIM + tx + j * BLOCK_SIZE;
                    int shared_row = ty + i * BLOCK_SIZE;
                    int shared_col = tx + j * BLOCK_SIZE;
                    float *dst = &A_shared[next_buf][shared_row][shared_col];
                    const float *src = A + global_row * K + global_col;
                    if (global_row < M && global_col < K) {
                        asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" 
                                      : 
                                      : "r"(dst), "l"(src), "n"(4));
                    } else {
                        *dst = 0.0f;
                    }
                }
            }
            // Load next tile for B into buffer next_buf
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int global_row = next_tile * TILE_DIM + ty + i * BLOCK_SIZE;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    int global_col = bx * TILE_DIM + tx + j * BLOCK_SIZE;
                    int shared_row = ty + i * BLOCK_SIZE;
                    int shared_col = tx + j * BLOCK_SIZE;
                    float *dst = &B_shared[next_buf][shared_row][shared_col];
                    const float *src = B + global_row * N + global_col;
                    if (global_row < K && global_col < N) {
                        asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" 
                                      : 
                                      : "r"(dst), "l"(src), "n"(4));
                    } else {
                        *dst = 0.0f;
                    }
                }
            }
            asm volatile ("cp.async.commit_group;");
        }

        // Wait for current tile's data to be ready before computation
        asm volatile ("cp.async.wait_group 0;");
        __syncthreads();

        // Compute the partial results using the data from buffer curr_buf
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = A_shared[curr_buf][ty][k];
            float a_val1 = A_shared[curr_buf][ty + BLOCK_SIZE][k];
            float b_val0 = B_shared[curr_buf][k][tx];
            float b_val1 = B_shared[curr_buf][k][tx + BLOCK_SIZE];
            Cvalue00 += a_val0 * b_val0;
            Cvalue01 += a_val0 * b_val1;
            Cvalue10 += a_val1 * b_val0;
            Cvalue11 += a_val1 * b_val1;
        }
        __syncthreads();

        // Swap the buffer for the next iteration
        curr_buf ^= 1;
    }

    // Write the computed 2x2 submatrix back to global memory with boundary checks
    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = Cvalue00;
    }
    if (row0 < M && (col0 + BLOCK_SIZE) < N) {
        C[row0 * N + col0 + BLOCK_SIZE] = Cvalue01;
    }
    if ((row0 + BLOCK_SIZE) < M && col0 < N) {
        C[(row0 + BLOCK_SIZE) * N + col0] = Cvalue10;
    }
    if ((row0 + BLOCK_SIZE) < M && (col0 + BLOCK_SIZE) < N) {
        C[(row0 + BLOCK_SIZE) * N + col0 + BLOCK_SIZE] = Cvalue11;
    }
}

// Host function to launch the asynchronous pipelined kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    async_pipeline_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Async pipelined matrix multiplication with cp.async (CUDA)");
}
