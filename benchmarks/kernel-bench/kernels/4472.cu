#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Compute statistics kernel using 2D grid indexing: grid.x = num_groups, grid.y = N
// Each block computes the mean and variance for one (n, g) group

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,              // product of dimensions from index 2 onward
    const int channels_per_group,   // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,    // output shape: (N, num_groups)
    scalar_t* __restrict__ var) {   // output shape: (N, num_groups)

  // Use 2D grid: blockIdx.y: sample index, blockIdx.x: group index
  int n = blockIdx.y;
  int g = blockIdx.x;

  // Compute starting offset for the current group in sample n
  int group_offset = n * C * spatial + g * channels_per_group * spatial;
  int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;

  // Each thread processes multiple elements via striding over group_elems
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
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
    // Write output for sample n and group g
    int out_index = n * num_groups + g;
    mean[out_index] = group_mean;
    var[out_index] = group_var;
  }
}

// Group Norm forward kernel using 2D grid indexing
// We map the (N, C) for each sample/channel pair to grid.y, and spatial dimension to grid.x
// Each block handles a tile of spatial elements

template <typename scalar_t>
__global__ void group_norm_forward_kernel(
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

  // grid.y covers the (n, c) pairs
  int nc = blockIdx.y;  // nc index: 0 <= nc < N * C
  int n = nc / C;
  int c = nc % C;

  // Determine group index from channel
  int g = c / channels_per_group;
  int stats_index = n * num_groups + g;
  scalar_t m = mean[stats_index];
  scalar_t v = var[stats_index];
  scalar_t inv_std = rsqrt(v + eps);

  // grid.x covers the spatial dimension in tiles
  int tile_index = blockIdx.x * blockDim.x + threadIdx.x;
  while (tile_index < spatial) {
    // Compute flattened index for (n, c, tile_index)
    int index = (n * C + c) * spatial + tile_index;
    scalar_t x_val = x[index];
    y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
    tile_index += blockDim.x * gridDim.x;
  }
}


// Host function that launches the kernels with appropriate 2D grid configurations
// and improved thread/block indexing

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
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Launch compute_stats_kernel using a 2D grid: (num_groups, N)
  int threads_stats = 256;
  dim3 blocks_stats(num_groups, N);
  size_t shared_mem_size = threads_stats * 2 * sizeof(float);

  // Launch group_norm_forward_kernel using a 2D grid: grid.x over spatial, grid.y over (N * C)
  int tile_dim = 256;
  int grid_x = (spatial + tile_dim - 1) / tile_dim;
  dim3 blocks_norm(grid_x, N * C);
  int threads_norm = tile_dim;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, stream>>>(
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with improved thread/block indexing");
}
