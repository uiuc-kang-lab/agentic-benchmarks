#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Use constant memory for weight and bias (assumes max channels <= 1024 and float type)
__constant__ float c_weight[1024];
__constant__ float c_bias[1024];

// Fused kernel: Computes per-group mean and variance and then normalizes the input
// in one pass. Each block processes one (sample, group) pair.

template <typename scalar_t>
__global__ void fused_group_norm_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  // Each block is mapped to one group for one sample (total blocks = N * num_groups)
  int group_idx = blockIdx.x;  // range: [0, N*num_groups)
  int n = group_idx / num_groups;
  int g = group_idx % num_groups;

  // Compute the starting offset for this group
  int group_offset = n * C * spatial + g * channels_per_group * spatial;
  int group_elems = channels_per_group * spatial;

  // Each thread computes a partial sum and sum of squares over its assigned elements
  scalar_t thread_sum = 0;
  scalar_t thread_sum_sq = 0;
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    // Determine local channel and spatial index
    int local_channel = i / spatial;  // index within the group
    int spatial_idx = i % spatial;
    int global_channel = g * channels_per_group + local_channel;
    int idx = group_offset + local_channel * spatial + spatial_idx;
    scalar_t val = x[idx];
    thread_sum += val;
    thread_sum_sq += val * val;
  }

  // Allocate shared memory for reduction (two arrays: one for sum and one for sum_sq)
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;

  s_sum[threadIdx.x] = thread_sum;
  s_sum_sq[threadIdx.x] = thread_sum_sq;
  __syncthreads();

  // Parallel reduction to compute total sum and sum of squares for the group
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Compute mean and variance (only thread 0 does this)
  __shared__ scalar_t shared_mean;
  __shared__ scalar_t shared_inv_std;
  if (threadIdx.x == 0) {
    shared_mean = s_sum[0] / static_cast<scalar_t>(group_elems);
    scalar_t var = s_sum_sq[0] / static_cast<scalar_t>(group_elems) - shared_mean * shared_mean;
    shared_inv_std = rsqrt(var + eps);
  }
  __syncthreads();

  // Apply normalization: y = (x - mean) * inv_std * weight + bias
  // Use constant memory for weight and bias
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    int local_channel = i / spatial;
    int spatial_idx = i % spatial;
    int global_channel = g * channels_per_group + local_channel;
    int idx = group_offset + local_channel * spatial + spatial_idx;
    scalar_t val = x[idx];
    y[idx] = (val - shared_mean) * shared_inv_std * c_weight[global_channel] + c_bias[global_channel];
  }
}

// Host launcher that copies weight and bias to constant memory and launches the fused kernel

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  // x shape: (N, C, ...)
  const int N = x.size(0);
  const int C = x.size(1);

  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  auto y = torch::empty_like(x);

  // Copy weight and bias to constant memory (assuming float type)
  cudaMemcpyToSymbol(c_weight, weight.data_ptr<float>(), C * sizeof(float));
  cudaMemcpyToSymbol(c_bias, bias.data_ptr<float>(), C * sizeof(float));

  // Each block works on one (n, g) group
  int total_groups = N * num_groups;
  // Choose number of threads per block. Use 256 or fewer if the group has fewer elements.
  int group_elems = channels_per_group * spatial;
  int threads = group_elems < 256 ? group_elems : 256;
  size_t shared_mem_size = threads * 2 * sizeof(float);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_group_norm_forward_cuda", ([&] {
    fused_group_norm_kernel<scalar_t><<<total_groups, threads, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
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
  m.def("forward", &group_norm_forward, "Fused Group Normalization forward (CUDA)");
}
