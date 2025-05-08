#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {
  
  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
    const int c = i / spatial;
    const int j = i % spatial;
    const scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + BLOCK_SIZE;
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const scalar_t group_mean = s_sum[0] / group_elems;
    const scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    mean[n * num_groups + g] = group_mean;
    var[n * num_groups + g] = group_var;
  }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void group_norm_forward_shared(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;

  __shared__ scalar_t sm_mean, sm_inv_std;
  if (threadIdx.x == 0) {
    const int stats_idx = n * num_groups + g;
    sm_mean = mean[stats_idx];
    sm_inv_std = rsqrt(var[stats_idx] + eps);
  }
  __syncthreads();

  const int group_elems = channels_per_group * spatial;
  for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
    const int c = i / spatial;
    const int j = i % spatial;
    const int linear_idx = n * C * spatial + (g * channels_per_group + c) * spatial + j;
    const scalar_t val = x[linear_idx];
    y[linear_idx] = (val - sm_mean) * sm_inv_std * weight[g * channels_per_group + c] + bias[g * channels_per_group + c];
  }
}

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); ++i) spatial *= x.size(i);
  const int cpg = C / num_groups;

  auto y = torch::empty_like(x);
  auto mean = torch::zeros({N, num_groups}, x.options());
  auto var = torch::zeros({N, num_groups}, x.options());

  const int stats_blocks = N * num_groups;
  const int stats_smem = 512 * 2 * x.element_size();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "compute_stats_kernel", ([&] { compute_stats_kernel<scalar_t, 512><<<stats_blocks, 512, stats_smem>>>(
      x.data_ptr<scalar_t>(), N, C, spatial, cpg, num_groups,
      mean.data_ptr<scalar_t>(), var.data_ptr<scalar_t>());

  group_norm_forward_shared<scalar_t, 512><<<stats_blocks, 512>>>(
      x.data_ptr<scalar_t>(),
      mean.data_ptr<scalar_t>(),
      var.data_ptr<scalar_t>(),
      weight.data_ptr<scalar_t>(),
      bias.data_ptr<scalar_t>(),
      N, C, spatial, cpg, num_groups,
      static_cast<scalar_t>(eps),
      y.data_ptr<scalar_t>());

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward, "GroupNorm optimized with shared memory");
}
