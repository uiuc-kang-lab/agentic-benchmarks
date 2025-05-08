#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_forward_kernel_shared(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  const int instance_idx = blockIdx.x;
  const int tid = threadIdx.x;
  
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;
  
  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;
  scalar_t* s_weight = reinterpret_cast<scalar_t*>(s_sum_sq + blockDim.x);
  scalar_t* s_bias = s_weight + normalized_size;

  // Cache weight and bias in shared memory
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    s_weight[i] = weight[i];
    s_bias[i] = bias[i];
  }
  __syncthreads();

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  const int vector_size = 4;
  const int aligned_size = normalized_size / vector_size * vector_size;
  
  // Vectorized accumulation
  #pragma unroll 2
  for (int i = tid * vector_size; i < aligned_size; i += blockDim.x * vector_size) {
    float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
    
    accscalar_t vals[4] = {
      static_cast<accscalar_t>(in_vec.x),
      static_cast<accscalar_t>(in_vec.y),
      static_cast<accscalar_t>(in_vec.z),
      static_cast<accscalar_t>(in_vec.w)
    };
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      local_sum += vals[j];
      local_sum_sq += vals[j] * vals[j];
    }
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

  // Warp-level reduction
  if (tid < 32) {
    volatile accscalar_t* vsum = s_sum;
    volatile accscalar_t* vsum_sq = s_sum_sq;
    if (blockDim.x >= 64) { vsum[tid] += vsum[tid + 32]; vsum_sq[tid] += vsum_sq[tid + 32]; }
    if (blockDim.x >= 32) { vsum[tid] += vsum[tid + 16]; vsum_sq[tid] += vsum_sq[tid + 16]; }
    if (blockDim.x >= 16) { vsum[tid] += vsum[tid + 8]; vsum_sq[tid] += vsum_sq[tid + 8]; }
    if (blockDim.x >= 8) { vsum[tid] += vsum[tid + 4]; vsum_sq[tid] += vsum_sq[tid + 4]; }
    if (blockDim.x >= 4) { vsum[tid] += vsum[tid + 2]; vsum_sq[tid] += vsum_sq[tid + 2]; }
    if (blockDim.x >= 2) { vsum[tid] += vsum[tid + 1]; vsum_sq[tid] += vsum_sq[tid + 1]; }
  }

  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Process output using cached weight and bias
  #pragma unroll 2
  for (int i = tid * vector_size; i < aligned_size; i += blockDim.x * vector_size) {
    float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
    float4 w_vec = *reinterpret_cast<const float4*>(&s_weight[i]);
    float4 b_vec = *reinterpret_cast<const float4*>(&s_bias[i]);
    
    float4 out_vec;
    accscalar_t vals[4] = {in_vec.x, in_vec.y, in_vec.z, in_vec.w};
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      accscalar_t norm_val = (static_cast<accscalar_t>(vals[j]) - mean) * inv_std;
      reinterpret_cast<scalar_t*>(&out_vec)[j] = 
        static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(reinterpret_cast<scalar_t*>(&w_vec)[j]) + 
                             static_cast<accscalar_t>(reinterpret_cast<scalar_t*>(&b_vec)[j]));
    }
    
    *reinterpret_cast<float4*>(&out_ptr[i]) = out_vec;
  }

  // Handle remaining elements using cached weight and bias
  for (int i = aligned_size + tid; i < normalized_size; i += blockDim.x) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(s_weight[i]) + 
                                      static_cast<accscalar_t>(s_bias[i]));
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  int threads = std::min(((normalized_size + 31) / 32) * 32, 1024);
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    // Additional shared memory for weight and bias caching
    int shared_size = threads * 2 * sizeof(accscalar_t) + 
                     2 * normalized_size * sizeof(scalar_t); // For weight and bias
    layernorm_forward_kernel_shared<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with shared memory caching",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}