#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Kernel 1: Compute partial sums and squares with minimal __syncthreads()
// This version unrolls the reduction manually to avoid excessive synchronizations.

template <typename scalar_t>
__global__ void compute_stats_kernel_atomic(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,                // product of dimensions from index 2 onward
    const int channels_per_group,       // C / num_groups
    const int num_groups,               // number of groups
    const int elements_per_block,       // number of elements processed per block
    scalar_t* __restrict__ global_sum,  // shape: (N, num_groups)
    scalar_t* __restrict__ global_sum_sq) { // shape: (N, num_groups)

  // Determine the sample index and group index from grid dimensions
  const int n = blockIdx.y;
  const int g = blockIdx.z;
  const int chunk = blockIdx.x;  // which chunk within the group

  const int group_size = channels_per_group * spatial;
  const int start = chunk * elements_per_block;
  int end = start + elements_per_block;
  if (end > group_size) end = group_size;

  // Compute base offset for this group
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;

  // Each thread computes a partial sum over its assigned elements
  scalar_t local_sum = 0;
  scalar_t local_sum_sq = 0;
  for (int idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
    scalar_t val = x[group_offset + idx];
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Shared memory for block reduction
  __shared__ scalar_t s_sum[256];
  __shared__ scalar_t s_sum_sq[256];

  s_sum[threadIdx.x] = local_sum;
  s_sum_sq[threadIdx.x] = local_sum_sq;
  __syncthreads(); // Ensure all threads have written their results

  // Manually unroll reduction to minimize __syncthreads()
  if (blockDim.x >= 256 && threadIdx.x < 128) {
    s_sum[threadIdx.x] += s_sum[threadIdx.x + 128];
    s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + 128];
  }
  __syncthreads();

  if (threadIdx.x < 64) {
    s_sum[threadIdx.x] += s_sum[threadIdx.x + 64];
    s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + 64];
  }
  __syncwarp(); // Lightweight warp-level synchronization

  if (threadIdx.x < 32) {
    volatile scalar_t* vs_sum = s_sum;
    volatile scalar_t* vs_sum_sq = s_sum_sq;
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 32];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 32];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 16];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 16];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 8];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 8];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 4];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 4];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 2];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 2];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 1];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 1];
  }

  // Only one thread performs the atomic addition
  if (threadIdx.x == 0) {
    int global_idx = n * num_groups + g;
    atomicAdd(&global_sum[global_idx], s_sum[0]);
    atomicAdd(&global_sum_sq[global_idx], s_sum_sq[0]);
  }
}

// Kernel 2: Finalize statistics by computing mean and variance from global accumulators
// Each thread handles one (n, group) pair.

template <typename scalar_t>
__global__ void finalize_stats_kernel_atomic(
    const scalar_t* __restrict__ global_sum,
    const scalar_t* __restrict__ global_sum_sq,
    const int group_size,
    scalar_t* __restrict__ mean,  // shape: (N, num_groups)
    scalar_t* __restrict__ var) { // shape: (N, num_groups)

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_groups = gridDim.x * blockDim.x;
  if (idx >= total_groups) return;

  scalar_t sum = global_sum[idx];
  scalar_t sum_sq = global_sum_sq[idx];
  scalar_t m = sum / static_cast<scalar_t>(group_size);
  scalar_t v = sum_sq / static_cast<scalar_t>(group_size) - m * m;
  mean[idx] = m;
  var[idx] = v;
}

// Kernel 3: Apply Group Normalization using the computed mean and variance
// Each thread processes one element of the input tensor.

template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,  // shape: (C)
    const scalar_t* __restrict__ bias,    // shape: (C)
    const int N,
    const int C,
    const int spatial,                // product of dimensions from index 2 onward
    const int channels_per_group,       // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * spatial;
  if (idx >= total) return;

  // Decode the flattened index into (n, c, j) coordinates
  int j = idx % spatial;
  int temp = idx / spatial;
  int c = temp % C;
  int n = temp / C;

  int g = c / channels_per_group;
  int stats_idx = n * num_groups + g;

  scalar_t m = mean[stats_idx];
  scalar_t v = var[stats_idx];
  scalar_t inv_std = rsqrt(v + eps);
  scalar_t x_val = x[idx];

  y[idx] = (x_val - m) * inv_std * weight[c] + bias[c];
}

// Host function: Group Normalization forward pass

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // Input tensor x has shape (N, C, ...), where ... are spatial dimensions
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }

  const int channels_per_group = C / num_groups;
  const int group_size = channels_per_group * spatial;

  // Launch parameters for compute_stats_kernel_atomic
  const int elements_per_block = 1024;  // Number of elements processed by one block
  int num_chunks = (group_size + elements_per_block - 1) / elements_per_block;
  // Grid dimensions: (num_chunks, N, num_groups)
  dim3 stats_grid(num_chunks, N, num_groups);
  int block_size = 256;

  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto global_sum = torch::zeros({N, num_groups}, options);
  auto global_sum_sq = torch::zeros({N, num_groups}, options);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_atomic_cuda", ([&] {
    compute_stats_kernel_atomic<scalar_t><<<stats_grid, block_size, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        elements_per_block,
        global_sum.data_ptr<scalar_t>(),
        global_sum_sq.data_ptr<scalar_t>());
  }));

  // Finalize statistics: each thread computes mean and variance for one (n, group) pair
  int total_groups = N * num_groups;
  int threads = 256;
  int blocks = (total_groups + threads - 1) / threads;

  auto mean_tensor = torch::empty({N, num_groups}, options);
  auto var_tensor = torch::empty({N, num_groups}, options);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "finalize_stats_kernel_atomic_cuda", ([&] {
    finalize_stats_kernel_atomic<scalar_t><<<blocks, threads, 0, stream>>>(
        global_sum.data_ptr<scalar_t>(),
        global_sum_sq.data_ptr<scalar_t>(),
        group_size,
        mean_tensor.data_ptr<scalar_t>(),
        var_tensor.data_ptr<scalar_t>());
  }));

  // Launch normalization kernel over all elements
  int total_elements = N * C * spatial;
  int norm_block_size = 256;
  int norm_blocks = (total_elements + norm_block_size - 1) / norm_block_size;
  auto y = torch::empty_like(x);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_kernel_atomic_cuda", ([&] {
    group_norm_forward_kernel<scalar_t><<<norm_blocks, norm_block_size, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        mean_tensor.data_ptr<scalar_t>(),
        var_tensor.data_ptr<scalar_t>(),
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with minimal synchronizations");
}
