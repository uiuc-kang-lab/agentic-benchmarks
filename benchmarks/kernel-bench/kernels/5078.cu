#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Optimized LayerNorm kernel combining 2D block indexing and warp shuffle reduction for efficiency

template <typename scalar_t>
__global__ void layernorm_forward_kernel_optimized(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;

  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }

  unsigned int mask = 0xffffffff;
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    local_sum    += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  unsigned int lane = tid & (warpSize - 1);
  unsigned int warpId = tid >> 5;

  __shared__ accscalar_t smem_sum[32];
  __shared__ accscalar_t smem_sum_sq[32];

  if (lane == 0) {
    smem_sum[warpId] = local_sum;
    smem_sum_sq[warpId] = local_sum_sq;
  }
  __syncthreads();

  accscalar_t block_sum = 0;
  accscalar_t block_sum_sq = 0;
  unsigned int num_warps = (nthreads + warpSize - 1) / warpSize;
  if (tid < num_warps) {
    block_sum = smem_sum[tid];
    block_sum_sq = smem_sum_sq[tid];
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      block_sum    += __shfl_down_sync(mask, block_sum, offset);
      block_sum_sq += __shfl_down_sync(mask, block_sum_sq, offset);
    }
    if (tid == 0) {
      smem_sum[0] = block_sum;
      smem_sum_sq[0] = block_sum_sq;
    }
  }
  __syncthreads();

  accscalar_t final_sum = smem_sum[0];
  accscalar_t final_sum_sq = smem_sum_sq[0];

  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = final_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = final_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for the optimized LayerNorm forward pass

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  int total_threads = (normalized_size < 1024) ? normalized_size : 1024;
  int block_x = 32;
  int block_y = (total_threads + block_x - 1) / block_x;
  dim3 block(block_x, block_y);
  dim3 grid(outer_size);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = total_threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_optimized<scalar_t><<<grid, block, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) optimized",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
