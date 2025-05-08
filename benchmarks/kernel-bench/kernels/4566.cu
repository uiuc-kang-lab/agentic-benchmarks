#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256
#define WARP_SIZE 32

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_reduced_sync(
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

  // Each thread accumulates its portion
  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  
  #pragma unroll 4
  for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // Shared memory for reduction
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + BLOCK_SIZE;
  
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();  // Single sync point for shared memory writes

  // Warp-level reduction first
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + offset];
    }
  }

  // Block-level reduction with minimal syncs
  if (BLOCK_SIZE > WARP_SIZE) {
    __syncthreads();  // Sync only if we need cross-warp communication
    
    if (threadIdx.x < WARP_SIZE) {
      for (int offset = WARP_SIZE; offset < BLOCK_SIZE; offset += WARP_SIZE) {
        if (threadIdx.x + offset < BLOCK_SIZE) {
          s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
          s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + offset];
        }
      }
    }
  }

  // First thread writes results
  if (threadIdx.x == 0) {
    scalar_t group_mean = s_sum[0] / group_elems;
    scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    int out_index = n * num_groups + g;
    mean[out_index] = group_mean;
    var[out_index] = group_var;
  }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void group_norm_forward_kernel_reduced_sync(
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
  const int total = N * C * spatial;
  if (index >= total) return;

  const int j = index % spatial;
  const int temp = index / spatial;
  const int c = temp % C;
  const int n = temp / C;
  
  const int g = c / channels_per_group;
  const int stats_index = n * num_groups + g;
  
  const scalar_t m = mean[stats_index];
  const scalar_t v = var[stats_index];
  const scalar_t inv_std = rsqrt(v + eps);
  const scalar_t x_val = x[index];
  y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
}

torch::Tensor group_norm_forward_reduced_sync(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  const int total_groups = N * num_groups;
  const int total_elements = N * C * spatial;
  const int blocks_norm = (total_elements + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_reduced_sync", ([&] {
    const size_t shared_mem_size = BLOCK_SIZE_STATS * 2 * sizeof(scalar_t);

    compute_stats_kernel_reduced_sync<scalar_t, BLOCK_SIZE_STATS><<<
        total_groups, BLOCK_SIZE_STATS, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N, C, spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    group_norm_forward_kernel_reduced_sync<scalar_t, BLOCK_SIZE_NORM><<<
        blocks_norm, BLOCK_SIZE_NORM, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        N, C, spatial,
        channels_per_group,
        num_groups,
        static_cast<scalar_t>(eps),
        y.data_ptr<scalar_t>());
  }));

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward_reduced_sync, "Reduced Sync Group Normalization forward (CUDA)");
}