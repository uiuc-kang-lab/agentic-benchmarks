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

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  int lane_id = tid % 32;
  int warp_id = tid / 32;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  // Calculate start and end indices for this thread
  int thread_start = tid * elements_per_thread;
  int thread_end = min(thread_start + elements_per_thread, normalized_size);

  // Process elements in chunks of 4 for better memory coalescing
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  #pragma unroll
  for (int idx = thread_start; idx < thread_end; idx += 4) {
    accscalar_t vals[4];
    #pragma unroll
    for (int j = 0; j < 4 && (idx + j) < thread_end; j++) {
      vals[j] = static_cast<accscalar_t>(in_ptr[idx + j]);
      local_sum += vals[j];
      local_sum_sq += vals[j] * vals[j];
    }
  }

  // Warp-level reduction
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
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
  if (tid < 32) {
    int num_warps = (blockDim.x + 31) / 32;
    local_sum = (tid < num_warps) ? s_sum[tid] : 0;
    local_sum_sq = (tid < num_warps) ? s_sum_sq[tid] : 0;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
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
  accscalar_t mean = s_sum[0] / normalized_size;
  accscalar_t variance = s_sum_sq[0] / normalized_size - mean * mean;
  accscalar_t inv_std = rsqrt(variance + eps);

  // Normalize and apply affine transformation
  // Each thread processes its assigned elements
  #pragma unroll
  for (int idx = thread_start; idx < thread_end; idx += 4) {
    #pragma unroll
    for (int j = 0; j < 4 && (idx + j) < thread_end; j++) {
      int current_idx = idx + j;
      accscalar_t val = static_cast<accscalar_t>(in_ptr[current_idx]);
      accscalar_t normalized = (val - mean) * inv_std;
      out_ptr[current_idx] = static_cast<scalar_t>(
          normalized * static_cast<accscalar_t>(weight[current_idx]) +
          static_cast<accscalar_t>(bias[current_idx]));
    }
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Calculate optimal thread and workload distribution
  const int max_threads = 256;  // Reduced from 1024 for better occupancy
  const int min_elements_per_thread = 16;
  
  int threads = min(max_threads, (normalized_size + min_elements_per_thread - 1) / min_elements_per_thread);
  threads = ((threads + 31) / 32) * 32;  // Round up to nearest multiple of warp size
  
  int elements_per_thread = (normalized_size + threads - 1) / threads;
  elements_per_thread = ((elements_per_thread + 3) / 4) * 4;  // Round up to multiple of 4 for vectorization

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    
    layernorm_forward_kernel_balanced<scalar_t><<<outer_size, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}