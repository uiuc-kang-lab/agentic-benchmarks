#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// CUDA kernel for Layer Normalization forward with optimized stride loops.
template <typename scalar_t>
__global__ void layernorm_forward_kernel_stride_optimized(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one instance along the outer dimension.
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  // Use the accumulation type for better precision.
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Allocate shared memory for partial sums and partial sums of squares.
  extern __shared__ char smem[];
  accscalar_t* s_sum    = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  // Process elements with a stride loop for coalesced memory access
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum    += val;
    local_sum_sq += val * val;
  }
  s_sum[tid]    = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Parallel reduction to compute the total sum and sum of squares using warp-level intrinsics for optimization.
  for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
    if (tid < stride) {
      s_sum[tid]    += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }
  if (tid < 32) {
    // Warp-level reduction; using volatile to avoid bank conflicts
    volatile accscalar_t* vs_sum = s_sum;
    volatile accscalar_t* vs_sum_sq = s_sum_sq;
    vs_sum[tid]    += vs_sum[tid + 32];
    vs_sum_sq[tid] += vs_sum_sq[tid + 32];
    vs_sum[tid]    += vs_sum[tid + 16];
    vs_sum_sq[tid] += vs_sum_sq[tid + 16];
    vs_sum[tid]    += vs_sum[tid + 8];
    vs_sum_sq[tid] += vs_sum_sq[tid + 8];
    vs_sum[tid]    += vs_sum[tid + 4];
    vs_sum_sq[tid] += vs_sum_sq[tid + 4];
    vs_sum[tid]    += vs_sum[tid + 2];
    vs_sum_sq[tid] += vs_sum_sq[tid + 2];
    vs_sum[tid]    += vs_sum[tid + 1];
    vs_sum_sq[tid] += vs_sum_sq[tid + 1];
  }

  // Compute mean and inverse standard deviation.
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation.
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) +
                                         static_cast<accscalar_t>(bias[i]));
  }
}

// C++ interface. Note: eps has a default value of 1e-5.
torch::Tensor layernorm_forward_stride_optimized(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create an output tensor with the same shape and options as x.
  auto output = torch::empty_like(x);

  // Determine normalized dimension size (the product of weight's dimensions)
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Choose number of threads (cap at 1024)
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_stride_optimized", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_stride_optimized<scalar_t><<<blocks, threads, shared_size>>>(
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
  // Bind the 'forward' function with default eps argument.
  m.def("forward", &layernorm_forward_stride_optimized, "LayerNorm forward with optimized stride (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
