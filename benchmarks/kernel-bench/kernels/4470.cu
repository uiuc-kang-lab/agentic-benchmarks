#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Kernel to compute per-group mean and variance using stride loops
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // Product of spatial dimensions
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // Output shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // Output shape: (N, num_groups)

  // Each block handles one (n, g) group
  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;

  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;

  // Use a stride loop so that threads cover all elements in the group
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    int c = i / spatial;
    int j = i % spatial;
    int index = group_offset + c * spatial + j;
    scalar_t val = x[index];
    sum += val;
    sum_sq += val * val;
  }

  // Reduction using shared memory
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scalar_t group_mean = s_sum[0] / group_elems;
    scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    mean[n * num_groups + g] = group_mean;
    var[n * num_groups + g] = group_var;
  }
}

// Kernel to apply group normalization using stride loops
// Each thread processes multiple elements if needed
template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,  // shape: (C)
    const scalar_t* __restrict__ bias,    // shape: (C)
    const int N,
    const int C,
    const int spatial,             // Product of spatial dimensions
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  const int total = N * C * spatial;
  // Use a stride loop to cover all elements
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

// Host function: Launches kernels with proper stride loops and boundary checks
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // x is expected to have shape (N, C, ...)
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

  // Launch configuration for compute_stats_kernel
  const int total_groups = N * num_groups;
  const int threads_stats = 256;
  const dim3 blocks_stats(total_groups);
  size_t shared_mem = 2 * threads_stats * sizeof(float);

  // Launch configuration for group_norm_forward_kernel
  const int total_elements = N * C * spatial;
  const int threads_norm = 256;
  const dim3 blocks_norm((total_elements + threads_norm - 1) / threads_norm);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with stride loops");
}
