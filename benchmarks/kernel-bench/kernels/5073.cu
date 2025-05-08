#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// This kernel minimizes warp divergence by using warp shuffle reductions with uniform control flow.
// The reduction is done in two stages: first, each warp reduces its own partial sums using unrolled, non-branching shuffle operations.
// Then, the warp-level results are reduced by a subset of threads in the first warp. Finally, the computed mean and inverse standard deviation
// are used to normalize the input. All operations maintain full precision.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_warpshuffle(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance (row) of the input
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each thread computes partial sums over elements assigned by striding through the normalized dimension
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum    += val;
    local_sum_sq += val * val;
  }

  // Warp-level reduction using shuffle intrinsic with uniform control flow
  unsigned int mask = 0xffffffff;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum    += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  // Get warp lane and warp id
  unsigned int lane = tid & (warpSize - 1);
  unsigned int warpId = tid >> 5;  // equivalent to tid / warpSize

  // Allocate shared memory for warp-level partial results
  __shared__ accscalar_t smem_sum[32];       // assume max 32 warps per block
  __shared__ accscalar_t smem_sum_sq[32];

  // The first lane of each warp stores its warp's reduction result
  if (lane == 0) {
    smem_sum[warpId] = local_sum;
    smem_sum_sq[warpId] = local_sum_sq;
  }
  __syncthreads();

  // Now, use the first warp to reduce the warp-level sums
  accscalar_t block_sum = 0;
  accscalar_t block_sum_sq = 0;
  // Calculate number of warps in the block
  unsigned int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  if (tid < num_warps) {
    block_sum = smem_sum[tid];
    block_sum_sq = smem_sum_sq[tid];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      block_sum    += __shfl_down_sync(mask, block_sum, offset);
      block_sum_sq += __shfl_down_sync(mask, block_sum_sq, offset);
    }
    // Only thread 0 gets the final result; store it back into shared memory
    if (tid == 0) {
      smem_sum[0] = block_sum;
      smem_sum_sq[0] = block_sum_sq;
    }
  }
  __syncthreads();

  // Final reduction results
  accscalar_t final_sum = smem_sum[0];
  accscalar_t final_sum_sq = smem_sum_sq[0];

  // Compute mean and inverse standard deviation in thread 0 and broadcast
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = final_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = final_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize the input and apply affine transformation with uniform control flow
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) + static_cast<accscalar_t>(bias[i]));
  }
}

// C++ interface function for LayerNorm forward

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Choose the number of threads. Use up to 1024 threads per block.
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    // No extra dynamic shared memory is needed since shared arrays are statically sized
    layernorm_forward_kernel_warpshuffle<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with warp shuffle reduction and minimized divergence",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
