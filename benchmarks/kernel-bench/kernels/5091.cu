#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_forward_kernel_warp_aligned(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  const int instance_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane_id = tid % warpSize;
  
  // Ensure pointers are aligned to 128-bit boundary
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;
  
  // Shared memory for warp-level reductions
  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x/warpSize;

  // Register variables for partial sums
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  // Process elements in chunks of float4 (16 bytes)
  constexpr int vector_size = 4;
  const int vectors_per_thread = (normalized_size / (blockDim.x * vector_size)) * vector_size;
  const int aligned_offset = vectors_per_thread * tid;
  
  // Main vectorized loop - perfectly coalesced accesses
  #pragma unroll 4
  for (int i = 0; i < vectors_per_thread; i += vector_size) {
    int idx = aligned_offset + i;
    float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[idx]);
    
    local_sum += in_vec.x + in_vec.y + in_vec.z + in_vec.w;
    local_sum_sq += in_vec.x * in_vec.x + in_vec.y * in_vec.y + 
                    in_vec.z * in_vec.z + in_vec.w * in_vec.w;
  }

  // Handle remaining elements with scalar operations
  const int remaining_start = vectors_per_thread * blockDim.x;
  for (int i = remaining_start + tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(__ldg(&in_ptr[i]));
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Warp-level reduction using shuffle operations
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
  }

  // First thread in each warp writes to shared memory
  if (lane_id == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction across warps
  if (tid < warpSize) {
    local_sum = (tid < blockDim.x/warpSize) ? s_sum[tid] : 0;
    local_sum_sq = (tid < blockDim.x/warpSize) ? s_sum_sq[tid] : 0;
    
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
  }

  // Compute statistics
  __shared__ accscalar_t mean, inv_std;
  if (tid == 0) {
    mean = local_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = local_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Vectorized normalization and affine transformation
  #pragma unroll 4
  for (int i = 0; i < vectors_per_thread; i += vector_size) {
    int idx = aligned_offset + i;
    float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[idx]);
    float4 w_vec = *reinterpret_cast<const float4*>(&weight[idx]);
    float4 b_vec = *reinterpret_cast<const float4*>(&bias[idx]);
    
    float4 out_vec;
    out_vec.x = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.x) - mean) * inv_std * 
                                     static_cast<accscalar_t>(w_vec.x) + static_cast<accscalar_t>(b_vec.x));
    out_vec.y = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.y) - mean) * inv_std * 
                                     static_cast<accscalar_t>(w_vec.y) + static_cast<accscalar_t>(b_vec.y));
    out_vec.z = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.z) - mean) * inv_std * 
                                     static_cast<accscalar_t>(w_vec.z) + static_cast<accscalar_t>(b_vec.z));
    out_vec.w = static_cast<scalar_t>((static_cast<accscalar_t>(in_vec.w) - mean) * inv_std * 
                                     static_cast<accscalar_t>(w_vec.w) + static_cast<accscalar_t>(b_vec.w));
    
    *reinterpret_cast<float4*>(&out_ptr[idx]) = out_vec;
  }

  // Handle remaining elements
  for (int i = remaining_start + tid; i < normalized_size; i += blockDim.x) {
    scalar_t val = __ldg(&in_ptr[i]);
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Use multiple of 32 threads for optimal warp utilization
  const int threads_per_block = 256;  // 8 warps per block
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = (threads_per_block/warpSize) * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_warp_aligned<scalar_t><<<blocks, threads_per_block, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with warp-aligned memory access",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}