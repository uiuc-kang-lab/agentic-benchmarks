#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_forward_kernel_balanced(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int elements_per_thread) {

  const int instance_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;
  
  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;
  
  // Compute starting position for this thread
  const int thread_start = tid * elements_per_thread;
  const int vector_size = 4;
  
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  // Process elements in chunks of 4 (float4) with stride optimization
  #pragma unroll 4
  for (int i = thread_start; i < thread_start + elements_per_thread; i += vector_size) {
    if (i + vector_size <= normalized_size) {
      float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
      
      accscalar_t vals[4];
      vals[0] = static_cast<accscalar_t>(in_vec.x);
      vals[1] = static_cast<accscalar_t>(in_vec.y);
      vals[2] = static_cast<accscalar_t>(in_vec.z);
      vals[3] = static_cast<accscalar_t>(in_vec.w);
      
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        local_sum += vals[j];
        local_sum_sq += vals[j] * vals[j];
      }
    }
    else if (i < normalized_size) {
      // Handle remaining elements
      for (int j = 0; j < vector_size && i + j < normalized_size; j++) {
        accscalar_t val = static_cast<accscalar_t>(__ldg(&in_ptr[i + j]));
        local_sum += val;
        local_sum_sq += val * val;
      }
    }
  }

  // First level reduction within warps
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
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
  if (tid < (blockDim.x / 32)) {
    local_sum = s_sum[tid];
    local_sum_sq = s_sum_sq[tid];
    
    #pragma unroll
    for (int offset = (blockDim.x / 64); offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    
    if (tid == 0) {
      s_sum[0] = local_sum;
      s_sum_sq[0] = local_sum_sq;
    }
  }
  __syncthreads();

  // Compute statistics
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Process output with balanced workload distribution
  #pragma unroll 4
  for (int i = thread_start; i < thread_start + elements_per_thread; i += vector_size) {
    if (i + vector_size <= normalized_size) {
      float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
      float4 w_vec = *reinterpret_cast<const float4*>(&weight[i]);
      float4 b_vec = *reinterpret_cast<const float4*>(&bias[i]);
      
      float4 out_vec;
      accscalar_t norm_vals[4];
      
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        accscalar_t val = (j == 0) ? in_vec.x : (j == 1) ? in_vec.y : (j == 2) ? in_vec.z : in_vec.w;
        accscalar_t w_val = (j == 0) ? w_vec.x : (j == 1) ? w_vec.y : (j == 2) ? w_vec.z : w_vec.w;
        accscalar_t b_val = (j == 0) ? b_vec.x : (j == 1) ? b_vec.y : (j == 2) ? b_vec.z : b_vec.w;
        
        norm_vals[j] = (static_cast<accscalar_t>(val) - mean) * inv_std;
        reinterpret_cast<scalar_t*>(&out_vec)[j] = 
            static_cast<scalar_t>(norm_vals[j] * static_cast<accscalar_t>(w_val) + static_cast<accscalar_t>(b_val));
      }
      
      *reinterpret_cast<float4*>(&out_ptr[i]) = out_vec;
    }
    else if (i < normalized_size) {
      for (int j = 0; j < vector_size && i + j < normalized_size; j++) {
        scalar_t in_val = __ldg(&in_ptr[i + j]);
        scalar_t w_val = __ldg(&weight[i + j]);
        scalar_t b_val = __ldg(&bias[i + j]);
        accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
        out_ptr[i + j] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w_val) + 
                                              static_cast<accscalar_t>(b_val));
      }
    }
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Calculate optimal thread block size and elements per thread
  const int max_threads = 256;  // Reduced from 1024 to allow more registers per thread
  const int min_elements_per_thread = 16;
  
  int threads = std::min(((normalized_size + min_elements_per_thread - 1) / min_elements_per_thread), max_threads);
  threads = ((threads + 31) / 32) * 32;  // Round up to nearest multiple of warp size
  
  int elements_per_thread = (normalized_size + threads - 1) / threads;
  elements_per_thread = ((elements_per_thread + 3) / 4) * 4;  // Round up to nearest multiple of vector size

  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_balanced<scalar_t><<<blocks, threads, shared_size>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        output.data_ptr<scalar_t>(),
        normalized_size,
        elements_per_thread);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) balanced",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}