#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Optimized LayerNorm kernel using warp shuffle reduction and minimal atomic operations in shared memory.
// Each block processes one instance from the outer dimension.

template <typename scalar_t>
__global__ void layernorm_forward_atomic_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Identify the instance for this block
  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  int laneId = tid % warpSize;

  const scalar_t* in_ptr = input + instance_idx * normalized_size;
  scalar_t* out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Declare static shared memory variables to hold block-level sums
  __shared__ accscalar_t block_sum;
  __shared__ accscalar_t block_sum_sq;
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;

  if (tid == 0) {
    block_sum = 0;
    block_sum_sq = 0;
  }
  __syncthreads();

  // Each thread accumulates partial sums for its assigned indices
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Use warp shuffle instructions for fast intra-warp reduction
  unsigned int mask = __activemask();
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  // Only the first thread of each warp writes its result to shared memory using atomicAdd
  if (laneId == 0) {
    atomicAdd(&block_sum, local_sum);
    atomicAdd(&block_sum_sq, local_sum_sq);
  }
  __syncthreads();

  // Thread 0 finalizes the mean and inverse standard deviation calculation
  if (tid == 0) {
    mean = block_sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = block_sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1.0) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize the input and apply the affine transformation
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
    accscalar_t norm_val = (val - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) +
                                         static_cast<accscalar_t>(bias[i]));
  }
}

// C++ interface that wraps the kernel launch
// Note: Using statically allocated shared memory so no dynamic shared size is needed.

torch::Tensor layernorm_forward_atomic(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Determine the number of threads (up to 1024) per block
  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_atomic_cuda", ([&] {
    layernorm_forward_atomic_kernel<scalar_t><<<blocks, threads>>>(
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
  m.def("forward", &layernorm_forward_atomic, "Optimized LayerNorm forward (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
