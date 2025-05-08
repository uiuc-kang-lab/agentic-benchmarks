#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Kernel to compute per-group mean and variance.
// Each block is assigned one group for one batch element.
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // shape: (N, num_groups)

  // Each block corresponds to one (n, g) pair.
  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  
  // Starting offset for group g of sample n.
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  // Each thread processes several elements via striding.
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    // Map flat index into channel and spatial indices.
    const int c = i / spatial;
    const int j = i % spatial;
    const scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // Use shared memory for block reduction.
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  // Standard parallel reduction.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // The first thread computes mean and variance.
  if (threadIdx.x == 0) {
    const scalar_t group_mean = s_sum[0] / group_elems;
    const scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    const int out_index = n * num_groups + g;
    mean[out_index] = group_mean;
    var[out_index] = group_var;
  }
}

// Kernel to apply the group normalization.
// Each thread processes one element from the input.
template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,  // shape: (C)
    const scalar_t* __restrict__ bias,    // shape: (C)
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C * spatial;
  if (index >= total) return;

  // Decode flattened index into (n, c, j) coordinates.
  const int j = index % spatial;
  const int temp = index / spatial;
  const int c = temp % C;
  const int n = temp / C;

  // Determine the group index for the channel.
  const int g = c / channels_per_group;
  const int stats_index = n * num_groups + g;
  const scalar_t m = mean[stats_index];
  const scalar_t v = var[stats_index];
  const scalar_t inv_std = rsqrt(v + eps);
  const scalar_t x_val = x[index];
  // Normalize and then apply affine transformation.
  y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
}

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // Expect x to have shape (N, C, *).
  const int N = x.size(0);
  const int C = x.size(1);

  // Compute spatial size as the product of dimensions from index 2 onward.
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  // Create an output tensor, same shape as x.
  auto y = torch::empty_like(x);

  // Allocate temporary buffers for group mean and variance.
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Set up grid dimensions for the stats kernel.
  const int total_groups = N * num_groups;
  const int group_elems = channels_per_group * spatial;
  // Use a number of threads equal to the minimum of 256 and the number of group elements.
  const int threads_stats = (group_elems < 256 ? group_elems : 256);
  const dim3 blocks_stats(total_groups);

  // Set up grid dimensions for the normalization kernel.
  const int total_elements = N * C * spatial;
  const int threads_norm = 256;
  const dim3 blocks_norm((total_elements + threads_norm - 1) / threads_norm);

  // Get the current CUDA stream using the c10 namespace.
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    // Launch the kernel to compute means and variances.
    const size_t shared_mem_size = threads_stats * 2 * sizeof(scalar_t);
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    // Launch the kernel to perform group normalization.
    group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, 0, stream>>>(
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA)");
}