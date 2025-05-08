#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Define constant block size and elements per subblock
constexpr int BLOCK_SIZE = 256;
constexpr int ELEMENTS_PER_BLOCK = 1024;

// Kernel to compute partial sums and squares over a sub-block of a group
// Grid dimensions:
//   blockIdx.x = sample index (n) in [0, N)
//   blockIdx.y = group index (g) in [0, num_groups)
//   blockIdx.z = sub-block index (sub) in [0, num_subblocks)
// Each block processes a contiguous chunk of ELEMENTS_PER_BLOCK elements from the group

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const int group_size,         // = channels_per_group * spatial
    const int num_subblocks,      // = (group_size + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK
    scalar_t* __restrict__ partial_sums,   // shape: [N, num_groups, num_subblocks]
    scalar_t* __restrict__ partial_squares) { // shape: [N, num_groups, num_subblocks]

  // Map grid indices
  int n = blockIdx.x;         // sample index
  int g = blockIdx.y;         // group index
  int sub = blockIdx.z;       // sub-block index

  int tid = threadIdx.x;

  // Compute the starting position within the group for this sub-block
  int start = sub * ELEMENTS_PER_BLOCK;
  int end = start + ELEMENTS_PER_BLOCK;
  if (end > group_size) end = group_size;

  // Compute the starting offset for this group in the input
  // Group offset: skip n * (C * spatial) + g * (channels_per_group * spatial)
  int group_offset = n * C * spatial + g * group_size;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;

  // Loop over the assigned sub-block with stride BLOCK_SIZE
  for (int i = start + tid; i < end; i += BLOCK_SIZE) {
    int idx = group_offset + i;
    scalar_t val = x[idx];
    sum += val;
    sum_sq += val * val;
  }

  // Shared memory reduction with warp-level primitives for improved performance
  __shared__ scalar_t s_sum[BLOCK_SIZE];
  __shared__ scalar_t s_sum_sq[BLOCK_SIZE];
  s_sum[tid] = sum;
  s_sum_sq[tid] = sum_sq;
  __syncthreads();

  // Reduce in shared memory until the warp level
  for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  // Use warp shuffle for the final reduction when stride < 32
  if (tid < 32) {
    scalar_t sum_val = s_sum[tid];
    scalar_t sum_sq_val = s_sum_sq[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
      sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
      sum_sq_val += __shfl_down_sync(0xffffffff, sum_sq_val, offset);
    }
    if (tid == 0) {
      int index = n * (num_groups * num_subblocks) + g * num_subblocks + sub;
      partial_sums[index] = sum_val;
      partial_squares[index] = sum_sq_val;
    }
  }
}

// Kernel to finalize the statistics by reducing partial results.
// We'll launch this with a 1D grid over (N * num_groups) threads.
// Each thread reduces all sub-block partial sums for one (n, group) pair.

template <typename scalar_t>
__global__ void finalize_stats_kernel(
    const scalar_t* __restrict__ partial_sums,
    const scalar_t* __restrict__ partial_squares,
    const int N,
    const int num_groups,
    const int num_subblocks,
    const int group_size,
    scalar_t* __restrict__ mean,  // shape: [N, num_groups]
    scalar_t* __restrict__ var) { // shape: [N, num_groups]

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * num_groups;
  if (idx >= total) return;

  int n = idx / num_groups;
  int g = idx % num_groups;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  for (int sub = 0; sub < num_subblocks; ++sub) {
    int index = n * (num_groups * num_subblocks) + g * num_subblocks + sub;
    sum += partial_sums[index];
    sum_sq += partial_squares[index];
  }

  scalar_t m = sum / group_size;
  scalar_t v = sum_sq / group_size - m * m;
  int out_idx = n * num_groups + g;
  mean[out_idx] = m;
  var[out_idx] = v;
}

// Kernel to apply group normalization on the input tensor.
// This kernel uses a 1D grid mapping over all elements of x.

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

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * spatial;

  for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
    // Decode index into n, c, and spatial index j
    int j = i % spatial;
    int temp = i / spatial;
    int c = temp % C;
    int n = temp / C;

    int g = c / channels_per_group;
    int stats_idx = n * num_groups + g;
    scalar_t m = mean[stats_idx];
    scalar_t v = var[stats_idx];
    scalar_t inv_std = rsqrt(v + eps);
    scalar_t x_val = x[i];
    y[i] = (x_val - m) * inv_std * weight[c] + bias[c];
  }
}


// Forward function for Group Normalization

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  const int N = x.size(0);
  const int C = x.size(1);

  // Compute spatial dimensions (product of dimensions from index 2 onward)
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }

  const int channels_per_group = C / num_groups;
  const int group_size = channels_per_group * spatial;
  const int num_subblocks = (group_size + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

  // Allocate output tensor
  auto y = torch::empty_like(x);

  // Allocate temporary buffers for mean and variance
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Allocate buffers for partial sums; shape: [N, num_groups, num_subblocks]
  auto partial_sums = torch::empty({N, num_groups, num_subblocks}, options);
  auto partial_squares = torch::empty({N, num_groups, num_subblocks}, options);

  // Launch compute_stats_kernel with a 3D grid: (n, g, subblock)
  dim3 stats_grid(N, num_groups, num_subblocks);
  dim3 stats_block(BLOCK_SIZE);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    compute_stats_kernel<scalar_t><<<stats_grid, stats_block, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        group_size,
        num_subblocks,
        partial_sums.data_ptr<scalar_t>(),
        partial_squares.data_ptr<scalar_t>());

    // Launch finalize_stats_kernel with 1D grid over (N * num_groups) threads
    int total_groups = N * num_groups;
    int threads_final = BLOCK_SIZE;
    int blocks_final = (total_groups + threads_final - 1) / threads_final;
    finalize_stats_kernel<scalar_t><<<blocks_final, threads_final, 0, stream>>>(
        partial_sums.data_ptr<scalar_t>(),
        partial_squares.data_ptr<scalar_t>(),
        N,
        num_groups,
        num_subblocks,
        group_size,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    // Launch group normalization kernel with 1D grid over all elements
    int total_elements = N * C * spatial;
    int threads_norm = BLOCK_SIZE;
    int blocks_norm = (total_elements + threads_norm - 1) / threads_norm;

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
