#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>
#include <math.h>

// Device function to compute partial sums and sums of squares.
template <typename scalar_t>
__device__ void compute_partial_sums(
    const scalar_t* __restrict__ in_ptr,
    int tid,
    int normalized_size,
    int stride,
    at::acc_type<scalar_t, true>& local_sum,
    at::acc_type<scalar_t, true>& local_sum_sq) {

  using accscalar_t = at::acc_type<scalar_t, true>;

  local_sum = 0;
  local_sum_sq = 0;
  for (int i = tid; i < normalized_size; i += stride) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t a_val = static_cast<accscalar_t>(val);
    local_sum += a_val;
    local_sum_sq += a_val * a_val;
  }
}

// Device function to perform parallel reduction in shared memory.
template <typename accscalar_t>
__device__ void parallel_reduce(
    accscalar_t* s_sum,
    accscalar_t* s_sum_sq,
    int tid,
    int nthreads) {

  for (int stride = nthreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }
}

// Device function to compute mean and inverse standard deviation.
template <typename accscalar_t>
__device__ void compute_mean_inv_std(
    accscalar_t& mean,
    accscalar_t& inv_std,
    accscalar_t sum,
    accscalar_t sum_sq,
    int normalized_size,
    float eps) {

  mean = sum / static_cast<accscalar_t>(normalized_size);
  accscalar_t var = sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
  inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
}

// CUDA kernel for LayerNorm forward using modular device functions.
template <typename scalar_t>
__global__ void layernorm_forward_kernel_modular(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

  int instance_idx = blockIdx.x;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
  scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + nthreads;

  accscalar_t local_sum, local_sum_sq;
  compute_partial_sums(in_ptr, tid, normalized_size, nthreads, local_sum, local_sum_sq);

  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  parallel_reduce(s_sum, s_sum_sq, tid, nthreads);

  __shared__ accscalar_t mean;
  __shared__ accscalar_t inv_std;
  if (tid == 0) {
    compute_mean_inv_std(mean, inv_std, s_sum[0], s_sum_sq[0], normalized_size, eps);
  }
  __syncthreads();

  for (int i = tid; i < normalized_size; i += nthreads) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t norm_val = (static_cast<accscalar_t>(val) - mean) * inv_std;
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w) + static_cast<accscalar_t>(b));
  }
}

// C++ interface function for the LayerNorm forward pass.
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);

  int normalized_size = weight.numel();
  int outer_size = x.numel() / normalized_size;

  int threads = (normalized_size < 1024) ? normalized_size : 1024;
  int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    int shared_size = threads * 2 * sizeof(accscalar_t);
    layernorm_forward_kernel_modular<scalar_t><<<blocks, threads, shared_size>>>(
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
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with modular device functions",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
