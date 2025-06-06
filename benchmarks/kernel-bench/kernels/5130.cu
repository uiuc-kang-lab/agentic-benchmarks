#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Combined CUDA kernel for LayerNorm using warp shuffle reduction and stride loop optimization
// This kernel processes the input as a 2D tensor of shape (outer_size, normalized_size).
// Each block processes one instance along the outer dimension. 
template <typename scalar_t>
__global__ void layernorm_forward_kernel_combined(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  // Use accumulation type for full precision
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Declare shared memory for storing partial sums
  const int warpSize = 32;
  int warp_count = (blockDim.x + warpSize - 1) / warpSize;

  extern __shared__ char smem[];
  accscalar_t* s_sum   = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + warp_count;

  // Each thread computes a partial sum and sum-of-squares over its portion in chunks for better coalescing
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  const int CHUNK_SIZE = 4;  // Process 4 elements per thread at a time
  const int num_chunks = (normalized_size + blockDim.x * CHUNK_SIZE - 1) / (blockDim.x * CHUNK_SIZE);

  for (int chunk = 0; chunk < num_chunks; chunk++) {
    const int chunk_start = chunk * blockDim.x * CHUNK_SIZE + tid;
    
    // Load and process CHUNK_SIZE elements per thread
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

  // Intra-warp reduction using warp shuffle primitives
  const unsigned int full_mask = 0xffffffff;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum    += __shfl_down_sync(full_mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(full_mask, local_sum_sq, offset);
  }

  // Write the reduced result of each warp to shared memory (only lane 0 of each warp does this)
  if (lane == 0) {
    s_sum[warp_id] = local_sum;
    s_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction among warp leaders: assume warp_count <= warpSize so they are in one warp
  if (tid < warp_count) {
    local_sum = s_sum[tid];
    local_sum_sq = s_sum_sq[tid];
  }
  __syncthreads();

  // Reduce across the warp leaders using warp shuffle
  if (tid < warpSize) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      local_sum    += __shfl_down_sync(full_mask, local_sum, offset, warpSize);
      local_sum_sq += __shfl_down_sync(full_mask, local_sum_sq, offset, warpSize);
    }
    // Thread 0 computes mean and inv_std
    if (tid == 0) {
      mean = local_sum / static_cast<accscalar_t>(normalized_size);
      accscalar_t var = local_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
      inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
    }
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) +
                                        static_cast<accscalar_t>(bias[i]));
  }
}

// C++ interface. Note: eps has a default value of 1e-5.
torch::Tensor layernorm_forward_combined(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create an output tensor with the same shape as x
  auto output = torch::empty_like(x);
  
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;
  const int warpSize = 32;
  int warp_count = (threads + warpSize - 1) / warpSize;
  int shared_size = warp_count * 2 * sizeof(at::acc_type<float, true>);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_combined", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shm_size = warp_count * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_combined<scalar_t><<<blocks, threads, shm_size>>>(
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
  m.def("forward", &layernorm_forward_combined, "LayerNorm forward (CUDA) with combined optimizations",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
