#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <algorithm>

// Kernel to compute partial sums and partial sums of squares for a chunk of samples
// x: pointer to chunk of input of shape [chunk_size, C, spatial]
// chunk_size: number of samples in this chunk
// C: total channels, spatial: product of spatial dims
// channels_per_group: C / num_groups
// num_groups: number of groups
// elements_per_block: number of elements each block processes in the group
// partial_sums, partial_squares: output buffers of shape [chunk_size, num_groups, num_stats_blocks]

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int chunk_size,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const int elements_per_block,
    scalar_t* __restrict__ partial_sums,
    scalar_t* __restrict__ partial_squares) {

  int tid = threadIdx.x;
  int bid = blockIdx.x;         // Block index over blocks per group
  int n = blockIdx.y;           // Sample index in chunk [0, chunk_size)
  int g = blockIdx.z;           // Group index [0, num_groups)

  int group_size = channels_per_group * spatial;
  int start_idx = bid * elements_per_block;
  int group_offset = n * C * spatial + g * channels_per_group * spatial;

  __shared__ scalar_t s_sum[256];
  __shared__ scalar_t s_sum_sq[256];
  s_sum[tid] = 0;
  s_sum_sq[tid] = 0;

  for (int i = tid; i < elements_per_block && (start_idx + i) < group_size; i += blockDim.x) {
    int global_idx = group_offset + start_idx + i;
    scalar_t val = x[global_idx];
    s_sum[tid] += val;
    s_sum_sq[tid] += val * val;
  }
  __syncthreads();

  // Parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    // num_stats_blocks = gridDim.x
    int num_stats_blocks = gridDim.x;
    int partial_idx = n * (num_groups * num_stats_blocks) + g * num_stats_blocks + bid;
    partial_sums[partial_idx] = s_sum[0];
    partial_squares[partial_idx] = s_sum_sq[0];
  }
}

// Kernel to finalize computation of mean and variance for each group in the chunk
// partial_sums and partial_squares are of shape [chunk_size, num_groups, num_stats_blocks]
// mean and var are outputs of shape [chunk_size, num_groups]

template <typename scalar_t>
__global__ void finalize_stats_kernel(
    const scalar_t* __restrict__ partial_sums,
    const scalar_t* __restrict__ partial_squares,
    const int num_stats_blocks,
    const int group_size,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {
  // New kernel: grid.x = sample index, grid.y = group index
  int n = blockIdx.x;  // sample index in chunk
  int g = blockIdx.y;  // group index
  int tid = threadIdx.x;
  
  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  // Each thread accumulates a portion of the partial blocks
  for (int i = tid; i < num_stats_blocks; i += blockDim.x) {
    int idx = n * (gridDim.y * num_stats_blocks) + g * num_stats_blocks + i;
    sum += partial_sums[idx];
    sum_sq += partial_squares[idx];
  }
  
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;

  s_sum[tid] = sum;
  s_sum_sq[tid] = sum_sq;
  __syncthreads();

  // Parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sum_sq[tid] += s_sum_sq[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    scalar_t group_mean = s_sum[0] / static_cast<scalar_t>(group_size);
    scalar_t group_var = s_sum_sq[0] / static_cast<scalar_t>(group_size) - group_mean * group_mean;
    int out_idx = n * gridDim.y + g;  // gridDim.y == num_groups
    mean[out_idx] = group_mean;
    var[out_idx] = group_var;
  }
}

// Kernel to apply group normalization for a chunk
// x: input of shape [chunk_size, C, spatial]
// mean, var: computed statistics of shape [chunk_size, num_groups]
// weight, bias: affine parameters of shape [C]
// y: output of shape [chunk_size, C, spatial]

template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const int chunk_size,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ y) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = chunk_size * C * spatial;
  if (idx >= total) return;

  int j = idx % spatial;
  int temp = idx / spatial;
  int c = temp % C;
  int n = temp / C;  // local sample index in chunk
  int g = c / channels_per_group;

  int stats_idx = n * num_groups + g;
  scalar_t m = mean[stats_idx];
  scalar_t v = var[stats_idx];
  scalar_t inv_std = rsqrt(v + eps);

  // Compute output index relative to chunk
  int out_idx = n * C * spatial + c * spatial + j;
  scalar_t x_val = x[out_idx];
  y[out_idx] = (x_val - m) * inv_std * weight[c] + bias[c];
}

// Host function that pipelines group normalization over the batch dimension using multiple CUDA streams

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // x shape: [N, C, *]
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;
  const int group_size = channels_per_group * spatial;

  // Determine number of streams (pipeline chunks) over the batch dimension
  int num_streams = (N >= 2 ? 2 : 1);
  int chunk_size = (N + num_streams - 1) / num_streams;  // ceiling division for chunk size

  // Allocate output tensors
  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Partial buffers for stats: shape [N, num_groups, num_stats_blocks]
  int elements_per_block = 1024;
  int num_stats_blocks = (group_size + elements_per_block - 1) / elements_per_block;
  auto partial_sums = torch::empty({N, num_groups, num_stats_blocks}, options);
  auto partial_squares = torch::empty({N, num_groups, num_stats_blocks}, options);

  // Create CUDA streams
  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  const int block_size = 256;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    using scalar_t = decltype(x.data_ptr<float>()[0]);
    // Process each chunk in a separate stream for pipelining
    for (int i = 0; i < num_streams; i++) {
      int offset_n = i * chunk_size;
      if (offset_n >= N) break;
      int current_chunk = std::min(chunk_size, N - offset_n);

      // Compute pointer offsets for this chunk
      scalar_t* x_ptr = x.data_ptr<scalar_t>() + offset_n * C * spatial;
      scalar_t* y_ptr = y.data_ptr<scalar_t>() + offset_n * C * spatial;
      scalar_t* mean_ptr = mean.data_ptr<scalar_t>() + offset_n * num_groups;
      scalar_t* var_ptr = var.data_ptr<scalar_t>() + offset_n * num_groups;
      scalar_t* ps_ptr = partial_sums.data_ptr<scalar_t>() + offset_n * num_groups * num_stats_blocks;
      scalar_t* psq_ptr = partial_squares.data_ptr<scalar_t>() + offset_n * num_groups * num_stats_blocks;

      // Launch compute_stats_kernel
      {
        dim3 stats_blocks(num_stats_blocks, current_chunk, num_groups);
        dim3 stats_threads(block_size);
        size_t shared_mem = 0;
        compute_stats_kernel<scalar_t><<<stats_blocks, stats_threads, shared_mem, streams[i]>>>(
            x_ptr,
            current_chunk,
            C,
            spatial,
            channels_per_group,
            num_groups,
            elements_per_block,
            ps_ptr,
            psq_ptr);
      }
      
      // Launch finalize_stats_kernel to compute mean and variance
      {
        // Launch configuration: one block in x-dimension, with num_groups threads; y-dim covers current_chunk
        dim3 finalize_blocks(1, current_chunk);
        dim3 finalize_threads(num_groups);
        finalize_stats_kernel<scalar_t><<<finalize_blocks, finalize_threads, 0, streams[i]>>>(
            ps_ptr,
            psq_ptr,
            static_cast<int>(num_groups),
            static_cast<int>(num_stats_blocks),
            group_size,
            mean_ptr,
            var_ptr);
      }
      
      // Launch group_norm_forward_kernel to apply normalization
      {
        int total_elements = current_chunk * C * spatial;
        int norm_blocks = (total_elements + block_size - 1) / block_size;
        dim3 norm_grid(norm_blocks);
        group_norm_forward_kernel<scalar_t><<<norm_grid, block_size, 0, streams[i]>>>(
            x_ptr,
            current_chunk,
            C,
            spatial,
            channels_per_group,
            num_groups,
            static_cast<scalar_t>(eps),
            mean_ptr,
            var_ptr,
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            y_ptr);
      }
    } // end for each chunk
  }));

  // Synchronize and destroy streams
  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with pipelining");
}
