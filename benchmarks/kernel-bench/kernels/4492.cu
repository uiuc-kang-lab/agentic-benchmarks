#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256
#define WARP_SIZE 32

typedef float4 float4_t;

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N, 
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

  const int n = blockIdx.y;
  const int g = blockIdx.x;
  
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  const int vec_size = sizeof(float4_t) / sizeof(scalar_t);
  const int num_vectors = group_elems / vec_size;
  const int remaining = group_elems % vec_size;

  scalar_t thread_sum = 0;
  scalar_t thread_sum_sq = 0;

  const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
  for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    float4_t v = x_vec[i];
    thread_sum += v.x + v.y + v.z + v.w;
    thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }

  if (threadIdx.x < remaining) {
    int idx_rem = num_vectors * vec_size + threadIdx.x;
    scalar_t val = x[group_offset + idx_rem];
    thread_sum += val;
    thread_sum_sq += val * val;
  }

  thread_sum = warpReduceSum(thread_sum);
  thread_sum_sq = warpReduceSum(thread_sum_sq);

  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + (blockDim.x / WARP_SIZE);

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  if (lane_id == 0) {
    s_sum[warp_id] = thread_sum;
    s_sum_sq[warp_id] = thread_sum_sq;
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
    thread_sum = s_sum[threadIdx.x];
    thread_sum_sq = s_sum_sq[threadIdx.x];
  }

  if (warp_id == 0) {
    thread_sum = warpReduceSum(thread_sum);
    thread_sum_sq = warpReduceSum(thread_sum_sq);

    if (lane_id == 0) {
      scalar_t group_mean = thread_sum / group_elems;
      scalar_t group_var = thread_sum_sq / group_elems - group_mean * group_mean;
      mean[n * num_groups + g] = group_mean;
      var[n * num_groups + g] = group_var;
    }
  }
}

template <typename scalar_t>
__global__ void group_norm_forward_kernel(
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

  const int nc = blockIdx.y;
  const int n = nc / C;
  const int c = nc % C;
  const int g = c / channels_per_group;

  const scalar_t m = mean[n * num_groups + g];
  const scalar_t v = var[n * num_groups + g];
  const scalar_t inv_std = rsqrt(v + eps);
  const scalar_t w = weight[c];
  const scalar_t b = bias[c];

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
       idx < spatial; 
       idx += blockDim.x * gridDim.x) {
    const int index = (n * C + c) * spatial + idx;
    y[index] = (x[index] - m) * inv_std * w + b;
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
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  dim3 blocks_stats(num_groups, N);
  const int threads_stats = BLOCK_SIZE_STATS;
  const size_t shared_mem_size = (threads_stats / WARP_SIZE) * 2 * sizeof(float);

  const int spatial_blocks = (spatial + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;
  dim3 blocks_norm(spatial_blocks, N * C);
  const int threads_norm = BLOCK_SIZE_NORM;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N, C, spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, 0, stream>>>(
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