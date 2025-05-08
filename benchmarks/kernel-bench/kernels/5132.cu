#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_forward_kernel_hybrid(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + (blockDim.x / warpSize);

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  
  const int CHUNK_SIZE = 4;
  const int num_chunks = (normalized_size + blockDim.x * CHUNK_SIZE - 1) / (blockDim.x * CHUNK_SIZE);
  
  #pragma unroll
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    const int chunk_start = chunk * blockDim.x * CHUNK_SIZE + tid;
    
    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
      const int idx = chunk_start + i * blockDim.x;
      if (idx < normalized_size) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        local_sum += val;
        local_sum_sq += val * val;
      }
    }
  }

  const unsigned int full_mask = 0xffffffff;
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum    += __shfl_down_sync(full_mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(full_mask, local_sum_sq, offset);
  }

  if (lane == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  if (tid < warpSize && tid < (blockDim.x / warpSize)) {
    local_sum = s_sum[tid];
    local_sum_sq = s_sum_sq[tid];
    
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      local_sum    += __shfl_down_sync(full_mask, local_sum, offset);
      local_sum_sq += __shfl_down_sync(full_mask, local_sum_sq, offset);
    }
  }

  __shared__ accscalar_t mean, inv_std;
  if (tid == 0) {
    mean = local_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = local_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  #pragma unroll
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    const int chunk_start = chunk * blockDim.x * CHUNK_SIZE + tid;
    
    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
      const int idx = chunk_start + i * blockDim.x;
      if (idx < normalized_size) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        accscalar_t norm_val = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[idx]) +
                                           static_cast<accscalar_t>(bias[idx]));
      }
    }
  }
}

torch::Tensor layernorm_forward_hybrid(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;
  
  int threads = std::min(1024, ((normalized_size + 3) / 4 + 31) & ~31);
  int blocks = outer_size;
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_hybrid", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int warps_per_block = (threads + 31) / 32;
    int shared_size = warps_per_block * 2 * sizeof(accscalar_t);
    
    layernorm_forward_kernel_hybrid<scalar_t><<<blocks, threads, shared_size>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        output.data_ptr<scalar_t>(),
        normalized_size);
  }));

  return output;
}