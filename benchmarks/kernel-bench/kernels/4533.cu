#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Kernel 1: Optimized compute_stats_kernel with minimal __syncthreads()
// This kernel computes partial sums and squares for a chunk of a group using warp-level reduction
// and avoids excessive synchronizations by using warp shuffle intrinsics.

template <typename scalar_t>
__global__ void compute_stats_kernel_atomic_optimized(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,                // product of dimensions from index 2 onward
    const int channels_per_group,       // C / num_groups
    const int num_groups,               // number of groups
    const int elements_per_block,       // number of elements to process per block
    scalar_t* __restrict__ global_sum,  // shape: (N, num_groups)
    scalar_t* __restrict__ global_sum_sq) { // shape: (N, num_groups)

  // Determine sample index and group index from grid dimensions
  const int n = blockIdx.y;
  const int g = blockIdx.z;
  const int chunk = blockIdx.x;  // which chunk within this group

  const int group_size = channels_per_group * spatial;
  const int start = chunk * elements_per_block;
  int end = start + elements_per_block;
  if (end > group_size) end = group_size;

  // Calculate the base offset for the group within input tensor x
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;

  int tid = threadIdx.x;

  // Each thread computes its local sum and square sum over its portion of the chunk
  scalar_t local_sum = 0;
  scalar_t local_sum_sq = 0;
  for (int i = start + tid; i < end; i += blockDim.x) {
    scalar_t val = x[group_offset + i];
    local_sum += val;
    local_sum_sq += val * val;
  }

  // Perform warp-level reduction using shuffle intrinsics
  unsigned int mask = 0xffffffff;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_sum     += __shfl_down_sync(mask, local_sum, offset);
    local_sum_sq  += __shfl_down_sync(mask, local_sum_sq, offset);
  }

  // Get warp id and lane id
  int warp_id = tid / warpSize;
  int lane = tid % warpSize;

  // Shared memory for storing per-warp partial results (max 32 warps assumed)
  __shared__ scalar_t warp_sum[32];
  __shared__ scalar_t warp_sum_sq[32];

  // Only lane 0 of each warp writes its result.
  if (lane == 0) {
    warp_sum[warp_id] = local_sum;
    warp_sum_sq[warp_id] = local_sum_sq;
  }

  // Synchronize to ensure all warps have written their results
  __syncthreads();

  // First warp reduces the per-warp results. Only threads in warp 0 participate.
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  if (warp_id == 0) {
    scalar_t sum_val = (lane < num_warps) ? warp_sum[lane] : 0;
    scalar_t sum_sq_val = (lane < num_warps) ? warp_sum_sq[lane] : 0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      sum_val    += __shfl_down_sync(mask, sum_val, offset);
      sum_sq_val += __shfl_down_sync(mask, sum_sq_val, offset);
    }
    if (lane == 0) {
      int global_idx = n * num_groups + g;
      atomicAdd(&global_sum[global_idx], sum_val);
      atomicAdd(&global_sum_sq[global_idx], sum_sq_val);
    }
  }
}

// Kernel 2: Finalize stats by computing mean and variance from global accumulators
// Each thread handles one (n, g) pair.

template <typename scalar_t>
__global__ void finalize_stats_kernel_atomic(
    const scalar_t* __restrict__ global_sum,
    const scalar_t* __restrict__ global_sum_sq,
    const int group_size,
    scalar_t* __restrict__ mean, // output: (N, num_groups)
    scalar_t* __restrict__ var) { // output: (N, num_groups)

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = gridDim.x * blockDim.x; // total threads launched should cover (N * num_groups)
  // Alternatively, we assume idx < (N * num_groups) is ensured by the launcher.
  if (idx >= (gridDim.x * blockDim.x)) return; // Not used; proper launch parameters assumed

  // Here, each thread is responsible for one (n, g) pair. Compute n and g from idx
  // Instead, we launch with total threads = N * num_groups.
  // So directly, idx is the (n, g) index.
  scalar_t sum = global_sum[idx];
  scalar_t sum_sq = global_sum_sq[idx];
  scalar_t m = sum / static_cast<scalar_t>(group_size);
  scalar_t v = sum_sq / static_cast<scalar_t>(group_size) - m * m;
  mean[idx] = m;
  var[idx] = v;
}

// Kernel 3: Group normalization forward kernel
// Each thread normalizes one element of the input tensor x

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

  // Decode flattened index into (n, c, j) coordinates
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

// Main function: Launch the three kernels for Group Normalization

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // x is expected to have shape (N, C, ...), where ... are the spatial dimensions
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }

  const int channels_per_group = C / num_groups;
  const int group_size = channels_per_group * spatial;

  // Kernel launch parameters for compute_stats_kernel_atomic_optimized
  const int elements_per_block = 1024;  // Each block processes 1024 elements from the group
  int num_chunks = (group_size + elements_per_block - 1) / elements_per_block;
  // Grid dimensions: (num_chunks, N, num_groups)
  dim3 stats_grid(num_chunks, N, num_groups);
  int block_size = 256;

  // Allocate global accumulators for sum and sum of squares (initialized to 0)
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto global_sum = torch::zeros({N, num_groups}, options);
  auto global_sum_sq = torch::zeros({N, num_groups}, options);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_atomic_cuda", ([&] {
    compute_stats_kernel_atomic_optimized<scalar_t><<<stats_grid, block_size, 0, stream>>>(
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

  // Kernel 2: Finalize stats by computing mean and variance per group
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

  // Kernel 3: Apply Group Normalization using computed mean and variance
  const int total_elements = N * C * spatial;
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with optimized synchronization");
}
