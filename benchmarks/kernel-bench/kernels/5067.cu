#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// CUDA kernel for Layer Normalization forward with shared memory for weight and bias
// We regard the input tensor as 2D: (outer_size, normalized_size), where normalized_size equals the number of elements in weight (and bias).
// This kernel leverages shared memory to preload weight and bias arrays, reducing global memory accesses in the normalization loop.

template <typename scalar_t>
__global__ void layernorm_forward_shared_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance along the outer dimension.
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointers for the current instance
  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  // Use the accumulation type for improved precision during reduction
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Shared memory layout:
  // First: reduction arrays for s_sum and s_sum_sq, each of size num_threads
  // Then: shared copies of weight (normalized_size elements) and bias (normalized_size elements)
  int shmem_reduction_bytes = num_threads * 2 * sizeof(accscalar_t);

  extern __shared__ char smem[];
  accscalar_t* s_sum    = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + num_threads;
  
  // Pointers for shared weight and bias arrays
  scalar_t* s_weight = reinterpret_cast<scalar_t*>(smem + shmem_reduction_bytes);
  scalar_t* s_bias   = s_weight + normalized_size;

  // Load weight and bias into shared memory (cooperative loading by threads)
  for (int i = tid; i < normalized_size; i += num_threads) {
    s_weight[i] = weight[i];
    s_bias[i]   = bias[i];
  }
  __syncthreads();

  // Compute local sums and sums-of-squares for input elements
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += num_threads) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum    += val;
    local_sum_sq += val * val;
  }
  s_sum[tid]    = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Parallel reduction to compute total sum and sum of squares
  for (int stride = num_threads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid]    += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Compute mean and inverse standard deviation
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation using shared weight and bias data
  for (int i = tid; i < normalized_size; i += num_threads) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(s_weight[i]) +
                                        static_cast<accscalar_t>(s_bias[i]));
  }
}

// C++ interface function for LayerNorm forward with shared memory optimizations
// Note: eps has a default value of 1e-5.

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create an output tensor with the same shape and options as x
  auto output = torch::empty_like(x);

  // Determine the normalized dimension size (product of weight's dimensions)
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Choose number of threads (cap at 1024)
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    // Calculate shared memory size:
    // Reduction area: threads * 2 * sizeof(accscalar_t)
    // Plus shared weight and bias: 2 * normalized_size * sizeof(scalar_t)
    int shmem_bytes = threads * 2 * sizeof(accscalar_t) + 2 * normalized_size * sizeof(scalar_t);
    
    layernorm_forward_shared_kernel<scalar_t><<<blocks, threads, shmem_bytes>>>(
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
  // Bind the forward function with default eps argument.
  m.def("forward", &layernorm_forward, "LayerNorm forward with shared memory (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
