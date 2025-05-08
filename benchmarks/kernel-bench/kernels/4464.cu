#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Warp-level reduction using __shfl_down_sync
template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(__activemask(), val, offset);
    }
    return val;
}

// Optimized kernel to compute per-group mean and variance using warp-level reduction.
// Each block handles one (n, g) group.
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions starting from index 2
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // output shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // output shape: (N, num_groups)

  // Determine batch and group indices
  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;

  // Compute starting offset for group g of sample n
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;

  // Each thread iterates over its portion of group elements
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    const int c = i / spatial;
    const int j = i % spatial;
    const scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // First stage: Warp-level reduction within each warp
  sum = warpReduceSum(sum);
  sum_sq = warpReduceSum(sum_sq);

  // Allocate shared memory for inter-warp reduction
  extern __shared__ char smem[];
  // Number of warps in this block
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + num_warps;

  int lane = threadIdx.x & (warpSize - 1);
  int warp_id = threadIdx.x >> 5; // equivalent to threadIdx.x / warpSize

  // Each warp's lane 0 writes its reduced result to shared memory
  if (lane == 0) {
    s_sum[warp_id] = sum;
    s_sum_sq[warp_id] = sum_sq;
  }
  __syncthreads();

  // Second stage: First warp reduces the partial sums from all warps
  scalar_t block_sum = 0;
  scalar_t block_sum_sq = 0;
  if (threadIdx.x < num_warps) {
    block_sum = s_sum[threadIdx.x];
    block_sum_sq = s_sum_sq[threadIdx.x];
  } else {
    block_sum = 0;
    block_sum_sq = 0;
  }
  if (threadIdx.x < warpSize) {
    block_sum = warpReduceSum(block_sum);
    block_sum_sq = warpReduceSum(block_sum_sq);
  }

  // Thread 0 writes the final computed mean and variance
  if (threadIdx.x == 0) {
    const scalar_t group_mean = block_sum / group_elems;
    const scalar_t group_var = block_sum_sq / group_elems - group_mean * group_mean;
    const int out_index = n * num_groups + g;
    mean[out_index] = group_mean;
    var[out_index] = group_var;
  }
}

// Kernel to perform group normalization using computed mean and variance.
// Each thread processes one element from the input tensor.
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

  // Decode flattened index into (n, c, j) coordinates
  const int j = index % spatial;
  const int temp = index / spatial;
  const int c = temp % C;
  const int n = temp / C;

  // Determine the group index for the channel
  const int g = c / channels_per_group;
  const int stats_index = n * num_groups + g;
  const scalar_t m = mean[stats_index];
  const scalar_t v = var[stats_index];
  const scalar_t inv_std = rsqrt(v + eps);
  const scalar_t x_val = x[index];
  // Normalize and apply affine transformation
  y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
}

// Host function to launch the CUDA kernels for Group Normalization
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // Expect x to have shape (N, C, *)
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  // Create output tensor with the same shape as x
  auto y = torch::empty_like(x);

  // Temporary buffers for computed mean and variance
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Set up grid dimensions for the stats kernel
  const int total_groups = N * num_groups;
  const int group_elems = channels_per_group * spatial;
  const int threads_stats = (group_elems < 256 ? group_elems : 256);
  const dim3 blocks_stats(total_groups);

  // Set up grid dimensions for the normalization kernel
  const int total_elements = N * C * spatial;
  const int threads_norm = 256;
  const dim3 blocks_norm((total_elements + threads_norm - 1) / threads_norm);

  // Get the current CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    int num_warps = (threads_stats + 31) / 32;
    size_t shmem = num_warps * 2 * sizeof(scalar_t);

    // Launch kernel to compute group-level statistics with warp-level optimizations
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shmem, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    // Launch kernel to apply group normalization
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with warp-level optimization");
}
