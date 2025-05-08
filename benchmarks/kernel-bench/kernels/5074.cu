#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

// CUDA kernel for LayerNorm forward with optimized thread and block indexing.
// This version uses 2D grid and block dimensions to better map threads to the problem domain.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_optimized_indexing(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int outer_size) {

  // Each block processes a subset of the outer dimension and the normalized dimension.
  int outer_idx = blockIdx.y;
  int norm_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (norm_idx >= normalized_size || outer_idx >= outer_size) return;

  // Pointers to the start of this instance's data.
  const scalar_t* __restrict__ in_ptr = input + outer_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + outer_idx * normalized_size;

  // Use the accumulation type for better precision.
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each thread computes its element's contribution to the mean and variance.
  accscalar_t val = (norm_idx < normalized_size) ? static_cast<accscalar_t>(in_ptr[norm_idx]) : 0;
  accscalar_t local_sum = val;
  accscalar_t local_sum_sq = val * val;

  // Shared memory for partial sums and sums of squares.
  extern __shared__ accscalar_t sdata[];
  accscalar_t* s_sum = sdata;
  accscalar_t* s_sum_sq = sdata + blockDim.x;

  s_sum[threadIdx.x] = local_sum;
  s_sum_sq[threadIdx.x] = local_sum_sq;
  __syncthreads();

  // Parallel reduction for total sum and sum of squares within the block.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Compute mean and inverse standard deviation (only one thread does this).
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (threadIdx.x == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation.
  if (norm_idx < normalized_size) {
    accscalar_t norm_val = (val - mean) * inv_std;
    scalar_t w = __ldg(&weight[norm_idx]);
    scalar_t b = __ldg(&bias[norm_idx]);
    out_ptr[norm_idx] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for the LayerNorm forward pass.
// This function sets up kernel launch parameters and invokes the CUDA kernel.

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create an output tensor with the same shape and options as x.
  auto output = torch::empty_like(x);

  // Determine the size of the normalized dimension (product of weight dimensions).
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Configure a 2D grid and block dimensions for better mapping.
  dim3 threads(256);
  dim3 blocks((normalized_size + threads.x - 1) / threads.x, outer_size);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads.x * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_optimized_indexing<scalar_t><<<blocks, threads, shared_size>>>(
        x.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        static_cast<float>(eps),
        output.data_ptr<scalar_t>(),
        normalized_size,
        outer_size);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Bind the 'forward' function with default eps argument.
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with optimized indexing",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
