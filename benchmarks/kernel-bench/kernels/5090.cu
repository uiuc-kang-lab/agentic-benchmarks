#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// This kernel uses warp-level shuffle reduction to combine partial sums and sums of squares.
// Only warp leaders perform atomicAdd on shared memory, minimizing atomic usage and reducing contention.
// Each block processes one instance of the input tensor.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_atomic(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one outer instance
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  int total_threads = blockDim.x;

  // Pointers to the current instance's data
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each thread computes a partial sum and sum of squares in registers
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += total_threads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }

  // Perform warp-level reduction using shuffle operations
  unsigned int mask = 0xffffffff;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum   += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  // Use shared memory to accumulate warp-level results with minimal atomic operations
  __shared__ accscalar_t block_sum;
  __shared__ accscalar_t block_sum_sq;
  if (threadIdx.x == 0) {
    block_sum = 0;
    block_sum_sq = 0;
  }
  __syncthreads();

  int lane = threadIdx.x & (warpSize - 1);
  if (lane == 0) { // Each warp leader atomically adds its result
    atomicAdd(&block_sum, local_sum);
    atomicAdd(&block_sum_sq, local_sum_sq);
  }
  __syncthreads();

  // Compute mean and inverse standard deviation from the accumulated sums
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (threadIdx.x == 0) {
    mean = block_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = block_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation
  for (int i = tid; i < normalized_size; i += total_threads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}


// C++ interface: launch one block per instance
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Choose number of threads (max 1024 per block)
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_atomic", ([&] {
    layernorm_forward_kernel_atomic<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &layernorm_forward,
        "LayerNorm forward (CUDA) with warp shuffle reduction and minimal atomic operations",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
