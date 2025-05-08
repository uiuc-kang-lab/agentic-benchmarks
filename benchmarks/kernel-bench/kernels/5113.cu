#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <typename scalar_t>
__global__ void layernorm_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  int num_warps = blockDim.x / 32;
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + num_warps;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  accscalar_t warp_sum = warpReduceSum(local_sum);
  accscalar_t warp_sum_sq = warpReduceSum(local_sum_sq);

  if (tid % 32 == 0) {
    int warp_id = tid / 32;
    s_sum[warp_id] = warp_sum;
    s_sum_sq[warp_id] = warp_sum_sq;
  }
  __syncthreads();

  if (tid < 32) {
    accscalar_t block_sum = (tid < num_warps) ? s_sum[tid] : 0;
    accscalar_t block_sum_sq = (tid < num_warps) ? s_sum_sq[tid] : 0;

    accscalar_t total_sum = warpReduceSum(block_sum);
    accscalar_t total_sum_sq = warpReduceSum(block_sum_sq);

    if (tid == 0) {
      accscalar_t mean = total_sum / normalized_size;
      accscalar_t var = total_sum_sq / normalized_size - mean * mean;
      accscalar_t inv_std = 1.0 / sqrt(var + eps);

      for (int i = 0; i < normalized_size; ++i) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        accscalar_t norm_val = (val - mean) * inv_std;
        out_ptr[i] = static_cast<scalar_t>(norm_val * weight[i] + bias[i]);
      }
    }
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;
  
  // Optimize thread block size based on normalized_size
  const int max_threads = 256;  // Reduced from 1024 for better occupancy
  int threads = std::min(((normalized_size + 31) / 32) * 32, max_threads);
  
  // Calculate optimal grid size
  const int max_blocks_per_sm = 2048 / threads;  // Approximate blocks per SM
  int device_id;
  cudaGetDevice(&device_id);
  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
  int target_blocks_per_sm = max_blocks_per_sm;
  
  // Adjust grid size based on device capabilities
  int grid_size = std::min(outer_size, num_sms * target_blocks_per_sm);
  
  int num_warps = threads / 32;
  int shared_size = 2 * num_warps * sizeof(at::acc_type<float, true>);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    layernorm_forward_kernel<scalar_t><<<outer_size, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}