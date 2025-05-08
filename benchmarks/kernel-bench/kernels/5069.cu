#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int outer_size) {

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int instance_idx = bid % outer_size;
  const int chunk_size = (normalized_size + blockDim.x - 1) / blockDim.x;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  #pragma unroll
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Warp-level reduction first
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
  }

  // Write warp results to shared memory
  if (tid % warpSize == 0) {
    s_sum[tid/warpSize] = local_sum;
    s_sum_sq[tid/warpSize] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction with the first warp
  if (tid < warpSize) {
    local_sum = (tid < blockDim.x/warpSize) ? s_sum[tid] : 0;
    local_sum_sq = (tid < blockDim.x/warpSize) ? s_sum_sq[tid] : 0;
    
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
  }

  __shared__ accscalar_t mean, inv_std;
  if (tid == 0) {
    mean = local_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = local_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = rsqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  #pragma unroll
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) +
                                     static_cast<accscalar_t>(bias[i]));
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  const int threads = 256;
  const int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel<scalar_t><<<blocks, threads, shared_size>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        output.data_ptr<scalar_t>(),
        normalized_size,
        outer_size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}