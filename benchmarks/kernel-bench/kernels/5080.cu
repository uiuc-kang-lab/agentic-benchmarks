#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Optimized LayerNorm forward kernel that combines 2D thread indexing for improved parallelism
// with coalesced memory accesses for efficient global loads. Each block processes one instance
// of the input and performs a shared memory reduction to compute the mean and variance.

template <typename scalar_t>
__global__ void layernorm_forward_kernel_opt(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  // Each block processes one outer instance.
  int instance_idx = blockIdx.x;

  // Use 2D thread indexing to cover the normalized dimension flexibly.
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int nthreads = blockDim.x * blockDim.y;

  // Pointers to the start of this instance's data.
  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  // Each thread computes a partial sum and sum of squares over a strided range.
  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += nthreads) {
    // Use __ldg for read-only, coalesced global memory access
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }

  // Allocate shared memory for reduction: first part for partial sums, second for sum of squares.
  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + nthreads;

  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  // Perform parallel reduction in shared memory.
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Compute mean and inverse standard deviation in one thread, then broadcast.
  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  // Normalize and apply the affine transformation using coalesced global memory accesses.
  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for the optimized LayerNorm forward pass.
// This function sets up the grid and block dimensions and launches the CUDA kernel.

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  // Create an output tensor with the same shape as x
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  // Configure a 2D thread block: fix 32 threads in x-dimension and compute y-dimension accordingly.
  int total_threads = (normalized_size < 1024) ? normalized_size : 1024;
  int block_x = 32;
  int block_y = (total_threads + block_x - 1) / block_x;  // ceil division
  dim3 block(block_x, block_y);
  dim3 grid(outer_size);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda_opt", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = total_threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_opt<scalar_t><<<grid, block, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) optimized with 2D indexing and coalesced accesses",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
