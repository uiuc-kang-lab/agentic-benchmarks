#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/AccumulateType.h>
#include <math.h>

// CUDA kernel for LayerNorm forward with even workload distribution
// Each block processes one outer instance, and each thread is assigned a contiguous chunk
// of the normalized dimension based on the total number of threads. This ensures a balanced workload
// across threads to avoid underutilization and bottlenecks.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_even(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {
  // Each block handles one outer instance
  int instance_idx = blockIdx.x;

  // One-dimensional block indexing
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  // Compute the inclusive workload for each thread by evenly dividing the normalized dimension
  int chunk = (normalized_size + nthreads - 1) / nthreads;  // ceiling division
  int start = tid * chunk;
  int end = start + chunk;
  if (end > normalized_size) end = normalized_size;

  // Pointers to the current instance's data
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each thread computes its partial sum and sum of squares over its assigned range
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = start; i < end; i++) {
    // Use __ldg for read-only, coalesced global memory access
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }

  // Allocate shared memory for reduction: first part for partial sums, second for partial sum of squares
  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + nthreads;
  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Perform parallel reduction to compute total sum and sum of squares
  for (int stride = nthreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Compute mean and inverse standard deviation using the first thread
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Each thread normalizes its assigned range and applies element-wise affine transformation
  for (int i = start; i < end; i++) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for the LayerNorm forward pass using even workload distribution
// Each outer instance is handled by one block, with workload partitioned evenly among threads

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create output tensor with the same shape as x
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Determine the number of threads per block, ensuring we do not exceed 1024
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_even", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    // Each block: threads for reduction (partial sums and partial sums of squares)
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_even<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with even workload distribution",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
