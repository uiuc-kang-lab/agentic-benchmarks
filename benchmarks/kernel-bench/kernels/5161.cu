#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

constexpr int MAX_CONST_SIZE = 8192;
__constant__ float c_weights[MAX_CONST_SIZE];
__constant__ float c_biases[MAX_CONST_SIZE];

template <typename scalar_t>
__global__ void layernorm_constant_kernel(
    const scalar_t* __restrict__ input,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  __shared__ accscalar_t shared_mean;
  __shared__ accscalar_t shared_inv_std;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  unsigned int mask = 0xFFFFFFFF;
  int lane = tid & 31;
  for (int offset = 16; offset > 0; offset /= 2) {
    local_sum    += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  int warp_id = tid >> 5;
  int numWarps = (blockDim.x + 31) / 32;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + numWarps;

  if (lane == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  if (tid == 0) {
    accscalar_t total_sum = 0;
    accscalar_t total_sum_sq = 0;
    for (int i = 0; i < numWarps; ++i) {
      total_sum += s_sum[i];
      total_sum_sq += s_sum_sq[i];
    }
    accscalar_t mean = total_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = (total_sum_sq / normalized_size) - (mean * mean);
    shared_mean = mean;
    shared_inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - shared_mean) * shared_inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(c_weights[i]) + 
                                      static_cast<accscalar_t>(c_biases[i]));
  }
}

torch::Tensor layernorm_constant_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps=1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  
  TORCH_CHECK(normalized_size <= MAX_CONST_SIZE, "Constant memory size exceeded");
  cudaMemcpyToSymbol(c_weights, weight.data_ptr<float>(), normalized_size * sizeof(float));
  cudaMemcpyToSymbol(c_biases, bias.data_ptr<float>(), normalized_size * sizeof(float));

  int outer_size = x.numel() / normalized_size;
  int threads = normalized_size < 1024 ? normalized_size : 1024;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_constant_cuda", ([&] {
    int warpCount = (threads + 31) / 32;
    int shared_size = warpCount * 2 * sizeof(at::acc_type<scalar_t, true>);
    layernorm_constant_kernel<scalar_t><<<outer_size, threads, shared_size>>>(
        x.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        output.data_ptr<scalar_t>(),
        normalized_size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_constant_forward, "LayerNorm with constant memory (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
