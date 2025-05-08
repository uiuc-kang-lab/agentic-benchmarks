#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Define block sizes
#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256

// Kernel to compute per-group mean and variance using a stride loop internal to each block.
// Each block is responsible for one (n, g) pair where n is the sample index and g is the group index.

template <typename scalar_t>
__global__ void compute_stats_kernel_stride(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,            // product of dimensions from index 2 onward
    const int channels_per_group, // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,  // output: shape (N, num_groups)
    scalar_t* __restrict__ var) { // output: shape (N, num_groups)

  // Each block processes one group for one sample
  const int idx = blockIdx.x; // idx in [0, N * num_groups)
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  // Stride loop: each thread processes multiple elements if needed
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // Use shared memory to reduce partial sums
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  // Parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
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

// Kernel to perform group normalization using a grid-stride loop to cover all elements
// in the input tensor. This ensures correct boundary handling for workloads larger than
// the number of threads available.

template <typename scalar_t>
__global__ void group_norm_forward_kernel_stride(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int N,
    const int C,
    const int spatial,            // product of dimensions from index 2 onward
    const int channels_per_group, // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  int total = N * C * spatial;
  // Grid-stride loop: each thread processes multiple elements if necessary
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    int j = idx % spatial;
    int temp = idx / spatial;
    int c = temp % C;
    int n = temp / C;
    int g = c / channels_per_group;
    int stats_index = n * num_groups + g;
    scalar_t m = mean[stats_index];
    scalar_t v = var[stats_index];
    scalar_t inv_std = rsqrt(v + eps);
    scalar_t x_val = x[idx];
    y[idx] = (x_val - m) * inv_std * weight[c] + bias[c];
  }
}

// Host function that wraps the CUDA kernel launches
// It computes per-group statistics first, then performs group normalization using grid-stride loops

torch::Tensor group_norm_forward_stride(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  // x is expected to be of shape (N, C, ...). Compute N, C and spatial size.
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  // Allocate output tensor and temporary tensors for mean and variance.
  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Launch parameters for the stats kernel: one block per (n, g) pair
  const int total_groups = N * num_groups;
  int threads_stats = BLOCK_SIZE_STATS;
  
  // Calculate optimal number of blocks for normalization kernel based on SM count
  int device_id;
  cudaGetDevice(&device_id);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
  // Aim for 2-4 blocks per SM for good occupancy without excessive oversubscription
  const int optimal_blocks_per_sm = 3;
  int max_blocks_norm = sm_count * optimal_blocks_per_sm;
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_stride_cuda", ([&] {
    // Shared memory for stats kernel
    size_t shared_mem = threads_stats * 2 * sizeof(scalar_t);
    compute_stats_kernel_stride<scalar_t><<<
        total_groups, threads_stats, shared_mem, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    // Launch the normalization kernel using a grid-stride loop to cover all elements
    int total_elements = N * C * spatial;
    int blocks_norm = (total_elements + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;
    group_norm_forward_kernel_stride<scalar_t><<<
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
  m.def("forward", &group_norm_forward_stride, "Group Normalization forward with stride loops (CUDA)");
}
