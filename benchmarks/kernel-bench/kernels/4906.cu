#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel 1: Compute the L1 norm (sum of absolute values) for each row using a 2D grid.
// The grid is organized with gridDim.y = N (one row per y index) and gridDim.x = number of tiles
// to cover the columns. Each block computes the sum for its tile and then adds it atomically
// to a global row_sum buffer.
__global__ void l1_norm_sum_kernel(const float* __restrict__ x,
                                     float* __restrict__ row_sum,
                                     int N, int D) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (col < D) {
        val = fabsf(x[row * D + col]);
    }
    
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = val;
    __syncthreads();

    // Reduce the tile's values in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 of the block atomically adds its tile's sum to the row's total sum
    if (threadIdx.x == 0) {
        atomicAdd(&row_sum[row], sdata[0]);
    }
}

// Kernel 2: Normalize each element of x by the corresponding row's L1 norm.
// It uses the same 2D grid configuration so that each block handles a contiguous tile of columns.
__global__ void l1_norm_norm_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    const float* __restrict__ row_sum,
                                    int N, int D) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < D) {
        float norm = row_sum[row];
        // Prevent division by zero
        if (norm == 0.0f)
            norm = 1e-12f;
        out[row * D + col] = x[row * D + col] / norm;
    }
}

// The forward function allocates a temporary tensor for storing row sums and launches
// two kernels. The first kernel computes the L1 norm for each row distributed across
// multiple blocks, and the second kernel normalizes the input using those sums.

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Allocate a temporary tensor for the row sums, initializing all entries to zero
  auto row_sum = torch::zeros({N}, x.options());

  // Choose a block size and compute grid dimensions in the x (column) dimension.
  int threads = 256;
  int grid_x = (D + threads - 1) / threads;
  dim3 blocks(grid_x, N);

  // Launch the first kernel to compute the per-row L1 norms.
  size_t shared_mem_size = threads * sizeof(float);
  l1_norm_sum_kernel<<<blocks, threads, shared_mem_size>>>(
    x.data_ptr<float>(),
    row_sum.data_ptr<float>(),
    N, D
  );

  // Launch the second kernel to normalize each element by the computed L1 norm
  l1_norm_norm_kernel<<<blocks, threads>>>(
    x.data_ptr<float>(),
    out.data_ptr<float>(),
    row_sum.data_ptr<float>(),
    N, D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass optimized (CUDA)");
}
