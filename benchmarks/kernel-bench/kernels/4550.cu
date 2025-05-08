#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Tunable constant for threads per block in the stats kernel.
// We'll use 256 threads per block.
#define BLOCK_SIZE_STATS 256

// Kernel to compute per-group mean and variance using warp-level primitives for reduction.
// Each block processes one (n, g) pair.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_warp(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // shape: (N, num_groups)

  // Each block processes one (n, g) pair
  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;

  // Calculate group offset and number of elements in the group
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  // Each thread computes a partial sum and partial sum of squares
  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // Use warp-level reduction using __shfl_down_sync
  unsigned int mask = 0xffffffff; // full mask for 32 threads
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    sum    += __shfl_down_sync(mask, sum, offset);
    sum_sq += __shfl_down_sync(mask, sum_sq, offset);
  }

  // Each warp's lane 0 holds the reduced values
  int lane = threadIdx.x & (warpSize - 1);
  int warpId = threadIdx.x / warpSize;

  // Shared memory to store per-warp partial results
  __shared__ volatile scalar_t shared_sum[BLOCK_SIZE / 32];
  __shared__ volatile scalar_t shared_sum_sq[BLOCK_SIZE / 32];

  if (lane == 0) {
    shared_sum[warpId] = sum;
    shared_sum_sq[warpId] = sum_sq;
  }
  __syncthreads();

  // Let the first warp accumulate the results from all warps
  if (threadIdx.x < warpSize) {
    int numWarps = BLOCK_SIZE / 32;
    if (threadIdx.x < numWarps) {
      sum = shared_sum[threadIdx.x];
      sum_sq = shared_sum_sq[threadIdx.x];
    } else {
      sum = 0;
      sum_sq = 0;
    }
    
    // Warp-level reduction within the first warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
      sum    += __shfl_down_sync(mask, sum, offset);
      sum_sq += __shfl_down_sync(mask, sum_sq, offset);
    }

    if (threadIdx.x == 0) {
      scalar_t group_mean = sum / group_elems;
      scalar_t group_var = sum_sq / group_elems - group_mean * group_mean;
      int out_index = n * num_groups + g;
      mean[out_index] = group_mean;
      var[out_index] = group_var;
    }
  }
}

// Kernel to apply group normalization. Each thread processes one element from the input.
// This kernel remains unchanged from previous versions.

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

  int j = index % spatial;
  int temp = index / spatial;
  int c = temp % C;
  int n = temp / C;

  int g = c / channels_per_group;
  int stats_index = n * num_groups + g;
  scalar_t m = mean[stats_index];
  scalar_t v = var[stats_index];
  scalar_t inv_std = rsqrt(v + eps);
  scalar_t x_val = x[index];
  y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
}


// Host function that launches both kernels.
// It computes per-group statistics using the warp-level reduction kernel and then applies group normalization.

torch::Tensor group_norm_forward_optimized_warp(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // x is expected to be of shape (N, C, ...)
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  // Output tensor
  auto y = torch::empty_like(x);

  // Temporary tensors for mean and variance
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var  = torch::empty({N, num_groups}, options);

  // Launch configuration for the compute_stats kernel
  const int total_groups = N * num_groups;
  const int threads_stats = BLOCK_SIZE_STATS; // Use BLOCK_SIZE_STATS threads per block

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_optimized_warp_cuda", ([&] {
    // Shared memory size for the stats kernel: one scalar per thread per reduction array (2 arrays: sum and sum_sq)
    size_t shared_mem_size = BLOCK_SIZE_STATS * 2 * sizeof(scalar_t);
    compute_stats_kernel_warp<scalar_t, BLOCK_SIZE_STATS><<<
        total_groups, threads_stats, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    // Launch configuration for the normalization kernel
    const int total_elements = N * C * spatial;
    const int threads_norm = 256;
    const int blocks_norm = (total_elements + threads_norm - 1) / threads_norm;
    group_norm_forward_kernel<scalar_t><<<
        blocks_norm, threads_norm, 0, stream>>>(
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
  m.def("forward", &group_norm_forward_optimized_warp, "Group Normalization forward with warp-level reduction (CUDA)");
}
