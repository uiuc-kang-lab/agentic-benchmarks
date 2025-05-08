#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel prefetches the entire row into shared memory to reduce redundant global memory accesses.
// It computes the L1 norm (sum of abs values) and then normalizes the row.

__global__ void l1_norm_forward_kernel_opt(const float* __restrict__ x,
                                             float* __restrict__ out,
                                             int N,
                                             int D) {
  // Allocate shared memory: first D elements for row data, next blockDim.x elements for partial reduction sums.
  extern __shared__ float sdata[];
  float* row_data = sdata;                 // size = D floats
  float* partial_sum = sdata + D;          // size = blockDim.x floats

  int row = blockIdx.x;

  // Load the current row from global memory into shared memory.
  // Each thread loads a strided set of elements.
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    row_data[col] = x[row * D + col];
  }
  __syncthreads();

  // Compute partial sum of absolute values using the prefetched row data.
  float sum = 0.0f;
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    sum += fabsf(row_data[col]);
  }
  partial_sum[threadIdx.x] = sum;
  __syncthreads();

  // Perform reduction in shared memory to compute the total L1 norm of the row.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float total_sum = partial_sum[0];
  // Avoid division by zero
  if (threadIdx.x == 0 && total_sum == 0.0f) {
    total_sum = 1e-12f;
    partial_sum[0] = total_sum;
  }
  __syncthreads();
  total_sum = partial_sum[0];

  // Use the prefetched row data in shared memory for normalization.
  for (int col = threadIdx.x; col < D; col += blockDim.x) {
    out[row * D + col] = row_data[col] / total_sum;
  }
}

// The forward function prepares the kernel launch with an appropriate shared memory size
// that accommodates both the row data and the partial sum reduction buffer.

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this kernel.");
  x = x.contiguous();

  auto out = torch::empty_like(x);
  int N = x.size(0);
  int D = x.size(1);

  // Determine number of threads per block
  int threads = std::min<int>(1024, D);

  // Allocate shared memory: first D floats for the row, then 'threads' floats for reduction.
  int shared_mem_size = (D + threads) * sizeof(float);

  l1_norm_forward_kernel_opt<<<N, threads, shared_mem_size>>>(
      x.data_ptr<float>(),
      out.data_ptr<float>(),
      N,
      D
  );

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "L1 Normalization forward pass with shared memory prefetch (CUDA)");
}
