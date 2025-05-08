#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256
#define VECTOR_WIDTH 4

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_opt(
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

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scalar_t group_mean = s_sum[0] / group_elems;
    scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    int out_index = n * num_groups + g;
    mean[out_index] = group_mean;
    var[out_index] = group_var;
  }
}

template <typename scalar_t, int BLOCK_SIZE, int VEC>
__global__ void group_norm_forward_kernel_opt(
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
  const int group_elems = channels_per_group * spatial;
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;

  __shared__ scalar_t sm_m, sm_v;
  if (threadIdx.x == 0) {
    sm_m = mean[n * num_groups + g];
    sm_v = var[n * num_groups + g];
  }
  __syncthreads();
  
  scalar_t inv_std = rsqrt(sm_v + eps);
  const int vec_spatial = spatial / VEC;
  
  using VecT = thrust::array<scalar_t, VEC>;
  const int vec_offset = group_offset / VEC;
  const int vec_c = g * (channels_per_group / VEC);

  for (int i = threadIdx.x; i < (group_elems / VEC); i += BLOCK_SIZE) {
    const int c = (i / vec_spatial) + vec_c;
    const int s = i % vec_spatial;
    
    const VecT x_vec = reinterpret_cast<const VecT*>(x)[vec_offset + c * vec_spatial + s];
    const VecT w_vec = reinterpret_cast<const VecT*>(weight)[c];
    const VecT b_vec = reinterpret_cast<const VecT*>(bias)[c];
    
    scalar_t result[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) {
      scalar_t val = reinterpret_cast<const scalar_t*>(&x_vec)[v];
      result[v] = (val - sm_m) * inv_std * reinterpret_cast<const scalar_t*>(&w_vec)[v] 
                  + reinterpret_cast<const scalar_t*>(&b_vec)[v];
    }
    
    reinterpret_cast<VecT*>(y)[vec_offset + c * vec_spatial + s] = 
      *reinterpret_cast<VecT*>(result);
  }
}

torch::Tensor group_norm_forward_optimized(
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
  const int total_groups = N * num_groups;
  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_optimized_cuda", ([&] {
    const size_t shared_mem_size = BLOCK_SIZE_STATS * 2 * sizeof(scalar_t);
    compute_stats_kernel_opt<scalar_t, BLOCK_SIZE_STATS><<<
        total_groups, BLOCK_SIZE_STATS, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    const int vec_spatial = spatial / VECTOR_WIDTH;
    const int total_vec_elems = total_groups * (channels_per_group / VECTOR_WIDTH) * vec_spatial;
    const int blocks_norm = (total_vec_elems + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;
    
    group_norm_forward_kernel_opt<scalar_t, BLOCK_SIZE_NORM, VECTOR_WIDTH><<<
        blocks_norm, BLOCK_SIZE_NORM, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        static_cast<scalar_t>(eps),
        y.data_ptr<scalar_t>());
  }));

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward_optimized, "Optimized Group Normalization forward (CUDA)");
}