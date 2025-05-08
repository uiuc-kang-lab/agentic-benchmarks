#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Efficient LayerNorm forward kernel that combines modular design with coalesced memory accesses
// and warp-level reduction for improved performance.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_efficient(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block handles one instance (row) of the input tensor
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  // Pointers to the current instance data
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  // Use the accumulation type for better precision
  using accscalar_t = at::acc_type<scalar_t, true>;

  // Allocate shared memory for partial sums and partial sums-of-squares
  extern __shared__ char smem[];
  accscalar_t* s_sum    = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + nthreads;

  // Compute local partial sums with coalesced global memory loads via __ldg
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum    += a_val;
    local_sum_sq += a_val * a_val;
  }
  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Parallel reduction using shared memory with warp-level unrolling
  for (int stride = nthreads / 2; stride > 32; stride /= 2) {
    if (tid < stride) {
      s_sum[tid]    += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Warp-level reduction - use volatile pointers to avoid extra __syncthreads()
  if (tid < 32) {
    volatile accscalar_t* vs_sum = s_sum;
    volatile accscalar_t* vs_sum_sq = s_sum_sq;
    for (int stride = 32; stride > 0; stride /= 2) {
      vs_sum[tid]    += vs_sum[tid + stride];
      vs_sum_sq[tid] += vs_sum_sq[tid + stride];
    }
  }
  __syncthreads();

  // Compute mean and inverse standard deviation (only one thread does this)
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize input and apply the affine transformation
  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
    // Load weight and bias using __ldg for coalesced read
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for the LayerNorm forward pass
// Combines benefits from both kernel designs for better performance

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create output tensor with same options as x
  auto output = torch::empty_like(x);

  // Determine normalized dimension size and outer size
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Use up to 1024 threads per block
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_efficient<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) efficient combination with inline warp reduction",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
