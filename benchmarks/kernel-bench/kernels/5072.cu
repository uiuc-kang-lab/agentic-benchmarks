#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

// Constant memory for weight and bias (64KB limit)
__constant__ float c_weight[32768];  // 32K floats = 128KB
__constant__ float c_bias[32768];    // 32K floats = 128KB

template <typename scalar_t>
__global__ void layernorm_forward_kernel_constant(
    const scalar_t* __restrict__ input,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum    = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  
  // Use vectorized loads where possible
  #pragma unroll
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t val = static_cast<accscalar_t>(in_val);
    local_sum    += val;
    local_sum_sq += val * val;
  }
  s_sum[tid]    = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Warp-level reduction first
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    if (tid < offset) {
      s_sum[tid]    += s_sum[tid + offset];
      s_sum_sq[tid] += s_sum_sq[tid + offset];
    }
    __syncthreads();
  }

  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = rsqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Process elements using constant memory for weight and bias
  #pragma unroll
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
    // Access weight and bias from constant memory
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(c_weight[i]) + 
                                      static_cast<accscalar_t>(c_bias[i]));
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Copy weight and bias to constant memory
  TORCH_CHECK(normalized_size <= 32768, "normalized_size must be <= 32768 for constant memory usage");
  cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), normalized_size * sizeof(float));
  cudaMemcpyToSymbol(c_bias, bias.data_ptr<float>(), normalized_size * sizeof(float));

  const int threads = 256;
  const int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_constant<scalar_t><<<blocks, threads, shared_size>>>(
        x.data_ptr<scalar_t>(),
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