#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256

// Kernel to compute per-group mean and variance with coalesced memory accesses.
// Each block processes one (n, g) pair. Using a pointer offset to ensure that consecutive
// threads read consecutive memory locations.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_mem_aligned(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // shape: (N, num_groups)

  // Each block handles one (n, g) pair.
  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;

  // Compute group offset for sample n and group g
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  // Create a local pointer to the group data to improve memory coalescing
  const scalar_t* __restrict__ group_ptr = x + group_offset;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;

  // Loop with stride BLOCK_SIZE; consecutive threads read consecutive elements
  for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
    scalar_t val = group_ptr[i];
    sum += val;
    sum_sq += val * val;
  }

  // Use shared memory for reduction
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + BLOCK_SIZE;

  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  // Parallel reduction using shared memory and warp-level primitives
  // Reduce until 32 threads remain
  for (int stride = BLOCK_SIZE / 2; stride > 32; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Warp-level reduction (no __syncthreads needed within a warp)
  if (threadIdx.x < 32) {
    volatile scalar_t* vs_sum = s_sum;
    volatile scalar_t* vs_sum_sq = s_sum_sq;
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 32];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 16];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 8];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 4];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 2];
    vs_sum[threadIdx.x] += vs_sum[threadIdx.x + 1];

    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 32];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 16];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 8];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 4];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 2];
    vs_sum_sq[threadIdx.x] += vs_sum_sq[threadIdx.x + 1];
  }

  // The first thread writes the result
  if (threadIdx.x == 0) {
    scalar_t group_mean = s_sum[0] / group_elems;
    scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    int out_index = n * num_groups + g;
    mean[out_index] = group_mean;
    var[out_index] = group_var;
  }
}

// Kernel to apply group normalization with coalesced accesses for the output tensor.
// Global thread indices ensure that threads in a warp access consecutive memory addresses.

template <typename scalar_t, int BLOCK_SIZE>
__global__ void group_norm_forward_kernel_mem_aligned(
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

  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int total = N * C * spatial;
  if (index >= total) return;

  // Decode the flattened index into (n, c, j) coordinates
  int j = index % spatial;
  int temp = index / spatial;
  int c = temp % C;
  int n = temp / C;
  
  // Determine the group for channel c
  int g = c / channels_per_group;
  int stats_index = n * num_groups + g;

  scalar_t m = mean[stats_index];
  scalar_t v = var[stats_index];
  scalar_t inv_std = rsqrt(v + eps);
  scalar_t x_val = x[index];

  // Apply normalization and affine transformation
  y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
}

// Host function launching the kernels

torch::Tensor group_norm_forward_mem_aligned(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  // Expecting x to have shape (N, C, *) with contiguous memory
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  // Allocate output tensor
  auto y = torch::empty_like(x);

  // Allocate temporary storage for computed means and variances
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Determine grid dimensions
  const int total_groups = N * num_groups;  // one block per group sample
  const int threads_stats = BLOCK_SIZE_STATS;

  const int total_elements = N * C * spatial;
  const int blocks_norm = (total_elements + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_mem_aligned_cuda", ([&] {
    // Compute shared memory size for reduction
    size_t shared_mem_size = BLOCK_SIZE_STATS * 2 * sizeof(scalar_t);

    // Launch kernel to compute per-group statistics with improved memory coalescing
    compute_stats_kernel_mem_aligned<scalar_t, BLOCK_SIZE_STATS><<<
        total_groups, BLOCK_SIZE_STATS, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    // Launch kernel to perform group normalization
    group_norm_forward_kernel_mem_aligned<scalar_t, BLOCK_SIZE_NORM><<<
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
  m.def("forward", &group_norm_forward_mem_aligned, "GroupNorm forward with memory aligned global accesses (CUDA)");
}
