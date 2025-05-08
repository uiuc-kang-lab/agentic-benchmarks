#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Define tile dimensions for 2D grid mapping
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 8

// Kernel 1: Compute the L1 norm (sum of absolute values) for each row.
// This kernel uses a 2D grid and block. Each block processes a tile of the input matrix.
// Within each block, threads load elements from global memory, compute absolute values,
// and perform a reduction along the x-dimension (columns) using shared memory.
// The resulting partial sum for each row (within the block) is then atomically added to a global
// array 'row_norms' which accumulates the L1 norm for each row.

__global__ void compute_row_norm_kernel(const float* __restrict__ x, float* __restrict__ row_norms, int N, int D) {
  // 2D block indexing
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread loads one element (if in-bounds) and computes its absolute value
  float val = 0.0f;
  if(row < N && col < D) {
    val = fabsf(x[row * D + col]);
  }

  // Allocate shared memory for reduction; size = blockDim.x * blockDim.y
  extern __shared__ float sdata[];
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  // For threads out-of-bound in columns, store 0.
  sdata[tid] = (row < N && col < D) ? val : 0.0f;
  __syncthreads();

  // Reduction along the x-dimension within each row of the block
  // Only threads with the same threadIdx.y participate together
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      sdata[tid] += sdata[tid + offset];
    }
    __syncthreads();
  }

  // Thread with threadIdx.x == 0 writes the partial sum for this row from this block
  if (threadIdx.x == 0 && row < N) {
    atomicAdd(&row_norms[row], sdata[threadIdx.y * blockDim.x]);
  }
}

// Kernel 2: Normalize each element of the matrix using the computed L1 norms.
// Uses the same 2D grid mapping to cover the entire matrix.

__global__ void normalize_kernel(const float* __restrict__ x, float* __restrict__ out, const float* __restrict__ row_norms, int N, int D) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < D) {
    float norm = row_norms[row];
    // Avoid division by zero
    if (norm == 0.0f) norm = 1e-12f;
    out[row * D + col] = x[row * D + col] / norm;
  }
}

// Host function that launches the two kernels.
// First, it computes the L1 norm for each row using a 2D grid mapping with kernel 1,
// then it normalizes each element in kernel 2 using the computed norms.

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for L1 normalization.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Define block and grid dimensions for 2D mapping
  dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid((D + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (N + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

  // Allocate a tensor for row L1 norms and initialize to zero
  auto row_norms = torch::zeros({N}, x.options());

  // Launch Kernel 1: Compute partial L1 norms per row
  // Shared memory size: blockDim.x * blockDim.y * sizeof(float)
  compute_row_norm_kernel<<<grid, block, BLOCK_WIDTH * BLOCK_HEIGHT * sizeof(float)>>>(
    x.data_ptr<float>(),
    row_norms.data_ptr<float>(),
    N,
    D
  );

  // Launch Kernel 2: Normalize each element using the computed row norms
  normalize_kernel<<<grid, block>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    row_norms.data_ptr<float>(),
    N,
    D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Two-pass 2D L1 Normalization (CUDA)");
}
