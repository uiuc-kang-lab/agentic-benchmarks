#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size. Must be a multiple of warp size for best performance.
#define TILE_SIZE 32

// This kernel uses double buffering with asynchronous copies (cp.async) to overlap
// data movement (global memory to shared memory) with computation. It pipelines the
// loading of the next tile while computing the current tile.

// Note: This kernel assumes the GPU supports the cp.async instructions (e.g. NVIDIA H100)
// and is compiled with CUDA 12.2 or later.

__global__ void triangular_mm_kernel_async(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    // Declare double-buffered shared memory for tiles of A and B.
    __shared__ float shA[2][TILE_SIZE][TILE_SIZE];
    __shared__ float shB[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;
    // For lower triangular matrices, C[row,col] = 0 if row < col.
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;

    // Determine the range of tiles along the k-dimension.
    int t_start = col / TILE_SIZE;           // starting tile index
    int t_end = row / TILE_SIZE;               // ending tile index
    // We'll pipeline tiles from t_start to t_end.

    int current_buf = 0;
    int next_buf = 1;

    // --- Preload the first tile into shared memory (synchronously using cp.async) ---
    {
      int t = t_start;
      int a_col = t * TILE_SIZE + threadIdx.x;
      if (a_col < N && a_col <= row) {
          asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                        :
                        : "r"(&shA[current_buf][threadIdx.y][threadIdx.x]),
                          "l"(A + row * N + a_col),
                          "n"(4));
      } else {
          shA[current_buf][threadIdx.y][threadIdx.x] = 0.f;
      }

      int b_row = t * TILE_SIZE + threadIdx.y;
      if (b_row < N && b_row >= col) {
          asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                        :
                        : "r"(&shB[current_buf][threadIdx.y][threadIdx.x]),
                          "l"(B + b_row * N + col),
                          "n"(4));
      } else {
          shB[current_buf][threadIdx.y][threadIdx.x] = 0.f;
      }
      asm volatile ("cp.async.commit_group;");
      asm volatile ("cp.async.wait_group 0;");
      __syncthreads();
    }

    // --- Pipeline loop: for each tile except the last, prefetch next tile while computing current tile ---
    for (int t = t_start; t < t_end; t++) {
      int next_tile = t + 1;

      // Asynchronously prefetch the next tile into the buffer 'next_buf'.
      int a_col = next_tile * TILE_SIZE + threadIdx.x;
      if (a_col < N && a_col <= row) {
          asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                        :
                        : "r"(&shA[next_buf][threadIdx.y][threadIdx.x]),
                          "l"(A + row * N + a_col),
                          "n"(4));
      } else {
          shA[next_buf][threadIdx.y][threadIdx.x] = 0.f;
      }
      int b_row = next_tile * TILE_SIZE + threadIdx.y;
      if (b_row < N && b_row >= col) {
          asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                        :
                        : "r"(&shB[next_buf][threadIdx.y][threadIdx.x]),
                          "l"(B + b_row * N + col),
                          "n"(4));
      } else {
          shB[next_buf][threadIdx.y][threadIdx.x] = 0.f;
      }
      asm volatile ("cp.async.commit_group;");

      // Ensure that the current tile's data is ready before computing
      __syncthreads();

      // Compute on the current tile (tile index t).
      int k_tile_start = t * TILE_SIZE;
      int k_tile_end   = (t + 1) * TILE_SIZE;
      int k_begin = k_tile_start < col ? col : k_tile_start;
      int k_end   = k_tile_end > (row + 1) ? (row + 1) : k_tile_end;
      int effective_tile = k_end - k_begin;

      if (effective_tile == TILE_SIZE) {
          #pragma unroll
          for (int k = 0; k < TILE_SIZE; k++) {
              sum += shA[current_buf][threadIdx.y][k] * shB[current_buf][k][threadIdx.x];
          }
      } else {
          for (int k = k_begin; k < k_end; k++) {
              int local_k = k - t * TILE_SIZE;
              sum += shA[current_buf][threadIdx.y][local_k] * shB[current_buf][local_k][threadIdx.x];
          }
      }

      // Swap buffers: the next tile becomes the current one.
      current_buf = next_buf;
      next_buf = 1 - next_buf;

      // Wait for the asynchronous copies for the next tile to complete before next iteration.
      asm volatile ("cp.async.wait_group 0;");
      __syncthreads();
    }

    // --- Process the final tile (tile index t_end) which is already loaded in current_buf ---
    {
      int t = t_end;
      int k_tile_start = t * TILE_SIZE;
      int k_tile_end   = (t + 1) * TILE_SIZE;
      int k_begin = k_tile_start < col ? col : k_tile_start;
      int k_end   = k_tile_end > (row + 1) ? (row + 1) : k_tile_end;
      int effective_tile = k_end - k_begin;

      if (effective_tile == TILE_SIZE) {
          #pragma unroll
          for (int k = 0; k < TILE_SIZE; k++) {
              sum += shA[current_buf][threadIdx.y][k] * shB[current_buf][k][threadIdx.x];
          }
      } else {
          for (int k = k_begin; k < k_end; k++) {
              int local_k = k - t * TILE_SIZE;
              sum += shA[current_buf][threadIdx.y][local_k] * shB[current_buf][local_k][threadIdx.x];
          }
      }
    }

    __syncthreads();
    C[row * N + col] = sum;
}

// --- C++ Interface: Forward function ---

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Create a dedicated CUDA stream to enable overlap of kernel execution and other operations.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    triangular_mm_kernel_async<<<blocks, threads, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipeline async triangular matrix multiplication (CUDA) with overlap of memory transfers and computation");
}
