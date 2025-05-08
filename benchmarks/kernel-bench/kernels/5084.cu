#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/AccumulateType.h>
#include <math.h>

// CUDA kernel for LayerNorm forward with dynamic block size tuning.
// Each block processes one instance of the input. The block size is chosen from candidate sizes
// (32, 64, 128, 256, 512) based on the normalized dimension size, to maximize resource utilization.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_blocksize(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block computes one outer instance
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  // Pointers to this instance's data
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Shared memory allocation for reduction: first half for sum, second half for sum of squares
  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  // Each thread processes elements in a strided loop
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }

  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Apply normalization and the affine transformation
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface for the LayerNorm forward pass with dynamic block size tuning.
// This function selects an optimal block size from candidate sizes based on the normalized dimension size.

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Candidate block sizes
  int candidate_sizes[5] = {32, 64, 128, 256, 512};
  // If normalized_size is smaller than the smallest candidate, use normalized_size
  int optimal_block_size = (normalized_size < candidate_sizes[0]) ? normalized_size : candidate_sizes[0];
  for (int i = 0; i < 5; i++) {
    if (candidate_sizes[i] <= normalized_size) {
      optimal_block_size = candidate_sizes[i];
    }
  }
  
  int threads = optimal_block_size;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_blocksize", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_blocksize<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with dynamic block size tuning",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
