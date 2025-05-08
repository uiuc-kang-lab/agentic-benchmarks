#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Combined LayerNorm forward kernel: uses 2D block indexing for coalesced memory access
// and warp shuffle reduction to minimize divergence and shared memory usage.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_combined(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance (row)
  int instance_idx = blockIdx.x;

  // Pointer to the input and output for this instance
  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  // Flatten 2D thread index to 1D
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int nthreads = blockDim.x * blockDim.y;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each thread computes partial sums for mean and variance over the row using a strided loop
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += nthreads) {
    // Use __ldg for efficient, read-only coalesced loads
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }

  // Perform warp-level reduction using shuffle operations
  unsigned int lane = tid % warpSize;
  unsigned int warpId = tid / warpSize;
  
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum    += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
  }

  // Allocate shared memory for warp-level partial results; maximum warps assumed <= 32
  __shared__ accscalar_t warp_sum[32];
  __shared__ accscalar_t warp_sum_sq[32];

  // The first lane of each warp writes its sum into shared memory
  if (lane == 0) {
    warp_sum[warpId] = local_sum;
    warp_sum_sq[warpId] = local_sum_sq;
  }
  __syncthreads();

  // Let the first warp reduce the warp-level results
  int numWarps = (nthreads + warpSize - 1) / warpSize;
  if (tid < numWarps) {
    local_sum = warp_sum[tid];
    local_sum_sq = warp_sum_sq[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      local_sum    += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    // Thread 0 now computes the final mean and inverse standard deviation
    if (tid == 0) {
      accscalar_t mean = local_sum / static_cast<accscalar_t>(normalized_size);
      accscalar_t var = local_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
      accscalar_t inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
      // Overload shared memory arrays to broadcast the computed values
      warp_sum[0] = mean;
      warp_sum_sq[0] = inv_std;
    }
  }
  __syncthreads();

  // Broadcast the final computed mean and inv_std
  accscalar_t mean = warp_sum[0];
  accscalar_t inv_std = warp_sum_sq[0];

  // Normalize the row and apply the affine transformation
  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}


// C++ interface function for the combined LayerNorm forward pass
// Maps each outer instance to one block and uses a 2D thread layout with warp shuffle reduction

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Choose the number of threads: up to 1024 threads per block (using 2D blocks)
  int total_threads = (normalized_size < 1024) ? normalized_size : 1024;
  int block_x = 32;
  int block_y = (total_threads + block_x - 1) / block_x;  // Ceiling division
  dim3 block(block_x, block_y);
  dim3 grid(outer_size);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_combined", ([&] {
    layernorm_forward_kernel_combined<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        output.data_ptr<scalar_t>(),
        normalized_size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with combined warp shuffle and 2D indexing optimization",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
