#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

// Constant memory for weight and bias
__constant__ float const_weight[1024];
__constant__ float const_bias[1024];

// Optimized CUDA kernel for LayerNorm forward using constant memory for weight and bias.
template <typename scalar_t>
__global__ void layernorm_forward_kernel_constant(
    const scalar_t* __restrict__ input,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  const int instance_idx = blockIdx.x;
  const int tid = threadIdx.x;
  
  // Align pointers to 128-bit boundary
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;
  
  // Shared memory
  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  // Calculate number of float4 elements we can process
  const int vector_size = 4;
  const int aligned_size = normalized_size / vector_size * vector_size;
  
  // Process elements in chunks of 4 (float4) for coalesced and vectorized access
  for (int i = tid * vector_size; i < aligned_size; i += blockDim.x * vector_size) {
    float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
    accscalar_t val1 = static_cast<accscalar_t>(in_vec.x);
    accscalar_t val2 = static_cast<accscalar_t>(in_vec.y);
    accscalar_t val3 = static_cast<accscalar_t>(in_vec.z);
    accscalar_t val4 = static_cast<accscalar_t>(in_vec.w);
    local_sum += val1 + val2 + val3 + val4;
    local_sum_sq += val1 * val1 + val2 * val2 + val3 * val3 + val4 * val4;
  }

  // Handle remaining elements
  for (int i = aligned_size + tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(__ldg(&in_ptr[i]));
    local_sum += val;
    local_sum_sq += val * val;
  }

  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Parallel reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Compute statistics
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Process output in vectors of 4 when possible
  for (int i = tid * vector_size; i < aligned_size; i += blockDim.x * vector_size) {
    float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
    float4 out_vec;
    out_vec.x = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.x) - mean) * inv_std * 
                                     static_cast<accscalar_t>(const_weight[i]) + static_cast<accscalar_t>(const_bias[i]));
    out_vec.y = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.y) - mean) * inv_std * 
                                     static_cast<accscalar_t>(const_weight[i + 1]) + static_cast<accscalar_t>(const_bias[i + 1]));
    out_vec.z = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.z) - mean) * inv_std * 
                                     static_cast<accscalar_t>(const_weight[i + 2]) + static_cast<accscalar_t>(const_bias[i + 2]));
    out_vec.w = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.w) - mean) * inv_std * 
                                     static_cast<accscalar_t>(const_weight[i + 3]) + static_cast<accscalar_t>(const_bias[i + 3]));
    
    *reinterpret_cast<float4*>(&out_ptr[i]) = out_vec;
  }

  // Handle remaining elements
  for (int i = aligned_size + tid; i < normalized_size; i += blockDim.x) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(const_weight[i]) + static_cast<accscalar_t>(const_bias[i]));
  }
}

// C++ interface function for the LayerNorm forward pass.
torch::Tensor layernorm_forward_constant(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Copy weight and bias to constant memory
  cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), normalized_size * sizeof(float));
  cudaMemcpyToSymbol(const_bias, bias.data_ptr<float>(), normalized_size * sizeof(float));

  // Ensure thread count is multiple of warp size (32)
  int threads = std::min(((normalized_size + 31) / 32) * 32, 1024);
  int blocks = outer_size;

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
  m.def("forward", &layernorm_forward_constant, "LayerNorm forward (CUDA) with constant memory",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
