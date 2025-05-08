#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256

// Compute per-group mean and variance using __ldg() for read-only global memory loads.
// Assumes that the input tensor 'x' is 128-bit aligned.
template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_aligned(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // shape: (N, num_groups)

  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  
  // Use __ldg() for read-only loads from global memory
  for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
    int c = i / spatial;
    int j = i % spatial;
    int index = group_offset + c * spatial + j;
    scalar_t val = __ldg(&x[index]);
    sum += val;
    sum_sq += val * val;
  }
  
  // Reduction in shared memory
  extern __shared__ __align__(sizeof(scalar_t)) char smem[];
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

// Kernel to apply group normalization using __ldg() for optimized read-only global memory accesses.
template <typename scalar_t, int BLOCK_SIZE>
__global__ void group_norm_forward_kernel_aligned(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {
  
  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int total = N * C * spatial;
  if (index >= total) return;
  
  int j = index % spatial;
  int temp = index / spatial;
  int c = temp % C;
  int n = temp / C;
  int g = c / channels_per_group;
  int stats_index = n * num_groups + g;
  
  // Use __ldg() for all read-only accesses
  scalar_t m = __ldg(&mean[stats_index]);
  scalar_t v = __ldg(&var[stats_index]);
  scalar_t inv_std = rsqrt(v + eps);
  scalar_t x_val = __ldg(&x[index]);
  scalar_t w = __ldg(&weight[c]);
  scalar_t b = __ldg(&bias[c]);
  
  y[index] = (x_val - m) * inv_std * w + b;
}

// Host function to launch the kernels
torch::Tensor group_norm_forward_aligned(
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

  // One block per group for stats kernel
  const int total_groups = N * num_groups;
  const int threads_stats = BLOCK_SIZE_STATS;

  const int total_elements = N * C * spatial;
  const int blocks_norm = (total_elements + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_aligned_cuda", ([&] {
    size_t shared_mem_size = BLOCK_SIZE_STATS * 2 * sizeof(scalar_t);
    compute_stats_kernel_aligned<scalar_t, BLOCK_SIZE_STATS><<<
      total_groups, BLOCK_SIZE_STATS, shared_mem_size, stream>>>(
      x.data_ptr<scalar_t>(),
      N,
      C,
      spatial,
      channels_per_group,
      num_groups,
      mean.data_ptr<scalar_t>(),
      var.data_ptr<scalar_t>());

    group_norm_forward_kernel_aligned<scalar_t, BLOCK_SIZE_NORM><<<
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
  m.def("forward", &group_norm_forward_aligned, "GroupNorm forward with __ldg() and 128-bit aligned loads (CUDA)");
}
