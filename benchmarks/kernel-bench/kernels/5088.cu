#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

// CUDA kernel for LayerNorm forward using warp-level primitives for efficient reduction.
// The kernel performs warp reduction using __shfl_down_sync to minimize shared memory usage.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_warp(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block handles one instance.
  int instance_idx = blockIdx.x;

  // Use threadIdx.x as flat thread index within block, maximum 32 threads per warp
  int tid = threadIdx.x;

  // Pointers to the start of this instance's data.
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Local variables for accumulation.
  accscalar_t sum = 0;
  accscalar_t sum_sq = 0;

  // Accumulate sum and squared sum using warp-level reductions.
  for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    sum += a_val;
    sum_sq += a_val * a_val;
  }

  // Perform warp reduction for sum and sum_sq
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
  }

  // Single warp handles block-wide reduction
  __shared__ accscalar_t shared_sum;
  __shared__ accscalar_t shared_sum_sq;
  if (tid % 32 == 0) {
    shared_sum = sum;
    shared_sum_sq = sum_sq;
  }
  __syncthreads();

  // Compute mean and inverse std effectively when warp zero
  if (tid == 0) {
    sum = 0;
    sum_sq = 0;
    for (int id = 0; id < blockDim.x / 32; ++id) {
      sum += shared_sum;
      sum_sq += shared_sum_sq;
    }

    accscalar_t mean = sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    accscalar_t inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));

    shared_sum = mean;
    shared_sum_sq = inv_std;
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation.
  accscalar_t mean = shared_sum;
  accscalar_t inv_std = shared_sum_sq;
  for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for LayerNorm forward pass using warp primitives.
// This function sets up kernel launch parameters and invokes the CUDA kernel.

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create an output tensor with the same shape and options as x.
  auto output = torch::empty_like(x);

  // Determine the size of the normalized dimension (product of weight dimensions).
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Maximum 1024 threads per block, ensured by 32 warps per block
  int threads = 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    // launch the kernel with warp sync
    layernorm_forward_kernel_warp<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with warp-level primitives",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
