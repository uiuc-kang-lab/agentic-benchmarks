/*
 This CUDA implementation of LayerNorm combines the benefits of modular code structure and improved reduction efficiency via warp-level shuffle intrinsics.
 It computes partial sums and sums-of-squares per thread, then uses an in-warp reduction (via __shfl_down_sync) to minimize shared memory usage and synchronization overhead.
 Finally, only one warp further reduces the computed warp-level results to obtain the final mean and variance, and normalizes the input with affine scaling.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsics for both sum and sum of squares
template <typename accscalar_t>
__device__ __forceinline__ void warp_reduce(accscalar_t &sum, accscalar_t &sum_sq) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum    += __shfl_down_sync(mask, sum, offset);
        sum_sq += __shfl_down_sync(mask, sum_sq, offset);
    }
}

// Optimized LayerNorm forward kernel with warp-level reduction and modular design
template <typename scalar_t>
__global__ void layernorm_forward_kernel_optimized(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance (row) of the input
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  // Pointer to the current instance
  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each thread computes partial sum and sum of squares over its assigned elements
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }

  // In-warp reduction using shuffle intrinsics
  warp_reduce(local_sum, local_sum_sq);

  // Each warp’s lane 0 stores its sum in shared memory
  int warp_id = tid / warpSize;
  int lane = tid & (warpSize - 1);

  // Allocate shared memory: first part for warp sums, second for warp sums of squares
  extern __shared__ accscalar_t smem[]; 
  // Calculate number of warps per block
  int num_warps = (nthreads + warpSize - 1) / warpSize;
  accscalar_t* s_sum    = smem;                // size: num_warps
  accscalar_t* s_sum_sq = smem + num_warps;      // size: num_warps

  if (lane == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction: first warp aggregates the warp-level partials
  accscalar_t block_sum = 0;
  accscalar_t block_sum_sq = 0;
  if (tid < num_warps) {
    block_sum = s_sum[tid];
    block_sum_sq = s_sum_sq[tid];
  }
  __syncthreads();

  // Let the first warp perform a final in-warp reduction
  if (tid < warpSize) {
    // For threads beyond the available warp entries, initialize to zero
    if (tid >= num_warps) {
      block_sum = 0;
      block_sum_sq = 0;
    }
    warp_reduce(block_sum, block_sum_sq);
  }

  // Compute mean and inverse standard deviation in thread 0
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = block_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = block_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize each element and apply affine transformation
  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for the LayerNorm forward pass
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create an output tensor with the same shape and options as x
  auto output = torch::empty_like(x);

  // Determine the normalized dimension size and the number of instances
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Use up to 1024 threads per block
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  // Launch the kernel with shared memory allocated for warp-level reductions
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_optimized", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int num_warps = (threads + 31) / 32;
    int shared_size = num_warps * 2 * sizeof(accscalar_t); // two arrays: one for sums and one for sum of squares
    layernorm_forward_kernel_optimized<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) optimized with warp-level reduction",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
