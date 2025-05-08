#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Optimized CUDA kernel for LayerNorm forward using warp-level primitives to reduce warp divergence.

template <typename scalar_t>
__global__ void optimized_layernorm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance along the outer dimension.
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Static shared memory for broadcasting the computed mean and inverse std
  __shared__ accscalar_t shared_mean;
  __shared__ accscalar_t shared_inv_std;

  // Each thread computes partial sum and sum-of-squares over its strided load
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Use warp-level shuffle to reduce within each warp, minimizing divergent branching.
  unsigned int mask = 0xFFFFFFFF;
  int lane = tid & 31;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_sum    += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  // Compute warp id and number of warps in this block
  int warp_id = tid >> 5;  // equivalent to tid / 32
  int numWarps = (blockDim.x + 31) / 32;

  // Use dynamic shared memory to store partial sums from each warp
  extern __shared__ char smem[];
  accscalar_t* s_sum    = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + numWarps;

  // The first lane of each warp writes its result to shared memory
  if (lane == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction from warp-level partial sums performed by thread 0
  if (tid == 0) {
    accscalar_t total_sum = 0;
    accscalar_t total_sum_sq = 0;
    for (int i = 0; i < numWarps; i++) {
      total_sum    += s_sum[i];
      total_sum_sq += s_sum_sq[i];
    }
    accscalar_t mean = total_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = total_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    shared_mean = mean;
    shared_inv_std = static_cast<accscalar_t>(1) / __fsqrt_rn(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Each thread normalizes its chunk and applies the affine transformation
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - shared_mean) * shared_inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) + 
                                        static_cast<accscalar_t>(bias[i]));
  }
}

// Host function that launches the optimized kernel

torch::Tensor optimized_layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create output tensor with the same shape as x
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Choose the number of threads (up to 1024 per block)
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int warpCount = (threads + 31) / 32;
    int shared_size = warpCount * 2 * sizeof(accscalar_t);
    optimized_layernorm_forward_kernel<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &optimized_layernorm_forward, "Optimized LayerNorm forward (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
