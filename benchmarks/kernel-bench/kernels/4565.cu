#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Define constants
#define BLOCK_SIZE_ATOMIC 256
#define BLOCKS_PER_GROUP 4
#define BLOCK_SIZE_NORM 256

// Kernel 1: Compute partial sums using atomic operations.
// Each block processes a chunk of the elements for one group and atomically adds its result.

template <typename scalar_t, int BLOCK_SIZE, int BLOCKS_PER_GROUP>
__global__ void groupnorm_atomic_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of spatial dimensions
    const int channels_per_group,
    const int num_groups,
    const int group_elems,         // channels_per_group * spatial
    scalar_t* __restrict__ accum_sum,    // global accumulator, shape: (N * num_groups)
    scalar_t* __restrict__ accum_sum_sq) // global accumulator, shape: (N * num_groups)
{
    // Total number of groups = N * num_groups
    int total_groups = N * num_groups;
    // Map grid index to a specific group and block index within that group
    int global_group = blockIdx.x % total_groups;         // which group this block processes
    int block_in_group = blockIdx.x / total_groups;         // which block for this group

    // Determine sample and group indices
    int n = global_group / num_groups;
    int g = global_group % num_groups;

    // Compute the starting pointer for this group in input x
    int group_offset = n * C * spatial + g * channels_per_group * spatial;

    // Divide the group's elements among BLOCKS_PER_GROUP blocks
    int chunk_size = (group_elems + BLOCKS_PER_GROUP - 1) / BLOCKS_PER_GROUP;
    int start = block_in_group * chunk_size;
    int end = start + chunk_size;
    if (end > group_elems) end = group_elems;

    scalar_t sum = 0;
    scalar_t sum_sq = 0;

    // Each thread processes multiple elements within its assigned chunk
    for (int i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
        int c = i / spatial;
        int j = i % spatial;
        scalar_t val = x[group_offset + c * spatial + j];
        sum += val;
        sum_sq += val * val;
    }

    // Reduce within the block using shared memory
    __shared__ scalar_t shared_sum[BLOCK_SIZE];
    __shared__ scalar_t shared_sum_sq[BLOCK_SIZE];

    shared_sum[threadIdx.x] = sum;
    shared_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Atomically add the reduced result from this block to the global accumulators
    if (threadIdx.x == 0) {
        atomicAdd(&accum_sum[global_group], shared_sum[0]);
        atomicAdd(&accum_sum_sq[global_group], shared_sum_sq[0]);
    }
}

// Kernel 2: Finalize the computation of mean and variance for each group

template <typename scalar_t>
__global__ void groupnorm_finalize_stats_kernel(
    const int N,
    const int num_groups,
    const int group_elems,
    const scalar_t* __restrict__ accum_sum,
    const scalar_t* __restrict__ accum_sum_sq,
    scalar_t* __restrict__ mean,  // output: (N, num_groups)
    scalar_t* __restrict__ var) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_groups;
    if (idx >= total) return;
    
    scalar_t sum = accum_sum[idx];
    scalar_t sum_sq = accum_sum_sq[idx];
    scalar_t m = sum / group_elems;
    scalar_t v = sum_sq / group_elems - m * m;
    mean[idx] = m;
    var[idx] = v;
}

// Kernel 3: Apply group normalization using computed mean and variance

template <typename scalar_t, int BLOCK_SIZE>
__global__ void group_norm_forward_kernel_opt(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,  // shape: (C)
    const scalar_t* __restrict__ bias,    // shape: (C)
    const int N,
    const int C,
    const int spatial,             // product of spatial dimensions
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int total = N * C * spatial;
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

// Host function that wraps the kernel launches

torch::Tensor group_norm_forward_atomic_optimized(
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
  const int group_elems = channels_per_group * spatial;
  
  // Create the output tensor (same shape as x)
  auto y = torch::empty_like(x);
  
  // Allocate global accumulators for partial sums and initialize to zero
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto accum_sum = torch::zeros({N * num_groups}, options);
  auto accum_sum_sq = torch::zeros({N * num_groups}, options);
  
  // Allocate tensors for final mean and variance (shape: (N, num_groups))
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);
  
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  
  // Launch Kernel 1: Compute partial sums using atomic operations
  int total_groups = N * num_groups;
  int grid_atomic = total_groups * BLOCKS_PER_GROUP;  // each group is processed by BLOCKS_PER_GROUP blocks
  dim3 threads_atomic(BLOCK_SIZE_ATOMIC);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_atomic_stats_cuda", ([&] {
    groupnorm_atomic_stats_kernel<scalar_t, BLOCK_SIZE_ATOMIC, BLOCKS_PER_GROUP><<<
      grid_atomic, BLOCK_SIZE_ATOMIC, 0, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        group_elems,
        accum_sum.data_ptr<scalar_t>(),
        accum_sum_sq.data_ptr<scalar_t>());
  }));

  // Launch Kernel 2: Finalize statistics by computing mean and variance
  int threads_final = 256;
  int blocks_final = (total_groups + threads_final - 1) / threads_final;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_finalize_stats_cuda", ([&] {
    groupnorm_finalize_stats_kernel<scalar_t><<<
      blocks_final, threads_final, 0, stream>>>(
        N,
        num_groups,
        group_elems,
        accum_sum.data_ptr<scalar_t>(),
        accum_sum_sq.data_ptr<scalar_t>(),
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());
  }));

  // Launch Kernel 3: Apply group normalization using the computed statistics
  int total_elements = N * C * spatial;
  int blocks_norm = (total_elements + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_atomic_optimized_cuda", ([&] {
    group_norm_forward_kernel_opt<scalar_t, BLOCK_SIZE_NORM><<<
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
  m.def("forward", &group_norm_forward_atomic_optimized, "Group Normalization forward with atomic optimized stats (CUDA)");
}
