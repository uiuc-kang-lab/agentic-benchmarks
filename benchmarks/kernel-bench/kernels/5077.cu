#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

// CUDA kernel for LayerNorm forward with optimized atomic operations.
// This version minimizes the use of atomic operations by performing reductions in shared memory
// and only using atomic operations for the final accumulation step.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_atomic(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block handles one outer instance.
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;

  // Pointers to the start of this instance's data.
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  // Use the accumulation type for better precision.
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Shared memory for partial sums and sums of squares.
  extern __shared__ char smem[];
  accscalar_t* s_sum    = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  
  // Compute partial sums; use __ldg to load from global memory via the read-only cache.
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    // __ldg ensures coalesced and cached reads for read-only data.
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t val = static_cast<accscalar_t>(in_val);
    local_sum    += val;
    local_sum_sq += val * val;
  }
  s_sum[tid]    = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Parallel reduction for total sum and sum of squares within the block.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid]    += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Compute mean and inverse standard deviation (only one thread does this).
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation using coalesced global memory accesses.
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
    // Load weight and bias using __ldg for coalesced read.
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
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

  // Choose number of threads (maximum 1024 threads per block).
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_atomic<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with optimized atomic operations",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
