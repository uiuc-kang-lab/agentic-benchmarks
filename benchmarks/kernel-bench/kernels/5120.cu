#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_forward_kernel_coalesced(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  const int instance_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_size = 32;
  const int lane_id = tid % warp_size;
  const int warp_id = tid / warp_size;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  // First pass: compute mean
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  // Ensure all elements are processed
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Warp-level reduction
  #pragma unroll
  for (int offset = warp_size/2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
  }

  // Store warp results to shared memory
  if (lane_id == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction across warps
  if (tid < warp_size) {
    const int num_warps = blockDim.x / warp_size;
    if (tid < num_warps) {
      local_sum = s_sum[tid];
      local_sum_sq = s_sum_sq[tid];
    } else {
      local_sum = 0;
      local_sum_sq = 0;
    }

    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
  }

  __shared__ accscalar_t mean, inv_std;
  if (tid == 0) {
    mean = local_sum / normalized_size;
    // Compute variance in a numerically stable way
    accscalar_t var = fmax(local_sum_sq / normalized_size - mean * mean, 0.0f);
    inv_std = rsqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Second pass: normalize with coalesced access
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(
        norm_val * static_cast<accscalar_t>(weight[i]) +
        static_cast<accscalar_t>(bias[i]));
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  const int warp_size = 32;
  const int warps_per_block = 8;
  const int threads = warp_size * warps_per_block;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_coalesced<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}