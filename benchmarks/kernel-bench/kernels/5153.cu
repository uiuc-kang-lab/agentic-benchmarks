#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_optimal_block_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }
  // Warp-level reduction to decrease register pressure
  const unsigned int mask = 0xffffffff;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  // Write warp results to shared memory
  if ((tid & (warpSize - 1)) == 0) {
    s_sum[tid / warpSize] = local_sum;
    s_sum_sq[tid / warpSize] = local_sum_sq;
  }
  __syncthreads();

  int num_warps = blockDim.x / warpSize;
  if (tid < num_warps) {
    local_sum = s_sum[tid];
    local_sum_sq = s_sum_sq[tid];
    for (int offset = num_warps/2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(mask, local_sum, offset);
      local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
    }
  }
  __syncthreads();

  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0]/normalized_size - mean*mean;
    inv_std = rsqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = norm_val * weight[i] + bias[i];
  }
}

torch::Tensor layernorm_optimal_block(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps=1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  const int max_threads = 1024;
  int size = std::min(normalized_size, max_threads);
  int threads = 512;
  if (size < 512) threads = size >= 256 ? 256 : size >= 128 ? 128 : size >= 64 ? 64 : 32;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = 2 * threads * sizeof(accscalar_t);
    layernorm_optimal_block_kernel<scalar_t><<<outer_size, threads, shared_size>>>(
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
  m.def("forward", &layernorm_optimal_block, "LayerNorm optimal block (CUDA)",
      py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}