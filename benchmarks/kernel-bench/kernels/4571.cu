#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 128
#define BLOCK_SIZE_NORM 512

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_tuned(
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
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + BLOCK_SIZE;
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  // First reduce using shared memory until we have 32 values left
  for (int stride = BLOCK_SIZE / 2; stride >= 32; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Use warp-level reduction for the final 32 elements
  if (threadIdx.x < 32) {
    scalar_t sum_val = s_sum[threadIdx.x];
    scalar_t sum_sq_val = s_sum_sq[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
      sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
      sum_sq_val += __shfl_down_sync(0xffffffff, sum_sq_val, offset);
    }
    if (threadIdx.x == 0) {
      scalar_t group_mean = sum_val / group_elems;
      scalar_t group_var = sum_sq_val / group_elems - group_mean * group_mean;
      mean[n * num_groups + g] = group_mean;
      var[n * num_groups + g] = group_var;
    }
  }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void group_norm_forward_kernel_tuned(
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

  const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (index >= N * C * spatial) return;

  const int j = index % spatial;
  const int temp = index / spatial;
  const int c = temp % C;
  const int n = temp / C;

  const int g = c / channels_per_group;
  const scalar_t inv_std = rsqrt(var[n * num_groups + g] + eps);
  y[index] = (x[index] - mean[n * num_groups + g]) * inv_std * weight[c] + bias[c];
}

torch::Tensor group_norm_forward_tuned(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) spatial *= x.size(i);
  const int channels_per_group = C / num_groups;

  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  const int total_stats_blocks = N * num_groups;
  const int total_norm_blocks = (N * C * spatial + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_tuned", ([&] {
    compute_stats_kernel_tuned<scalar_t, BLOCK_SIZE_STATS><<<
        total_stats_blocks, BLOCK_SIZE_STATS, BLOCK_SIZE_STATS * 2 * sizeof(scalar_t), stream>>>(
        x.data_ptr<scalar_t>(),
        N, C, spatial, channels_per_group, num_groups,
        mean.data_ptr<scalar_t>(), var.data_ptr<scalar_t>());

    group_norm_forward_kernel_tuned<scalar_t, BLOCK_SIZE_NORM><<<
        total_norm_blocks, BLOCK_SIZE_NORM, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        N, C, spatial, channels_per_group, num_groups,
        static_cast<scalar_t>(eps),
        y.data_ptr<scalar_t>());
  }));

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward_tuned, "Tuned Block GroupNorm forward (CUDA)");
}
