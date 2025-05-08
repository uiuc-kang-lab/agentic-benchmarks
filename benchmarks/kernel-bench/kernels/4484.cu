#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Experiment with block sizes; we choose 512 threads per block to maximize occupancy
#define BLOCK_SIZE_STATS 512
#define BLOCK_SIZE_NORM 512

// Use vectorized loads for coalesced memory access
typedef float4 float4_t;

// Optimized warp reduction with no divergent branches
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Compute statistics kernel: calculates mean and variance for each group in each sample
// Each block is responsible for one (n, g) pair
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,            // product of dimensions from index 2 onward
    const int channels_per_group, // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,  // output: (N, num_groups)
    scalar_t* __restrict__ var) { // output: (N, num_groups)

  // Each block corresponds to a unique (n, g) pair
  const int idx = blockIdx.x; 
  const int n = idx / num_groups;
  const int g = idx % num_groups;

  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  // Use float4 vectorized loads when possible
  const int vec_size = sizeof(float4_t) / sizeof(scalar_t);
  const int num_vectors = group_elems / vec_size;
  const int remaining = group_elems % vec_size;

  scalar_t thread_sum = 0;
  scalar_t thread_sum_sq = 0;

  // Process data in vectorized loads
  const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
  for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    float4_t v = x_vec[i];
    thread_sum += v.x + v.y + v.z + v.w;
    thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }

  // Process remaining elements if any
  if (threadIdx.x < remaining) {
    int idx_rem = num_vectors * vec_size + threadIdx.x;
    scalar_t val = x[group_offset + idx_rem];
    thread_sum += val;
    thread_sum_sq += val * val;
  }

  // Warp-level reduction
  thread_sum = warpReduceSum(thread_sum);
  thread_sum_sq = warpReduceSum(thread_sum_sq);

  // Shared memory for inter-warp reduction. Allocate one entry per warp.
  extern __shared__ char smem[];
  scalar_t* s_sum    = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + (blockDim.x / warpSize);

  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  if (lane_id == 0) {
    s_sum[warp_id] = thread_sum;
    s_sum_sq[warp_id] = thread_sum_sq;
  }
  __syncthreads();

  // Final reduction: first warp aggregates results from each warp
  if (warp_id == 0 && lane_id < ((blockDim.x + warpSize - 1) / warpSize)) {
    thread_sum = s_sum[lane_id];
    thread_sum_sq = s_sum_sq[lane_id];
    thread_sum = warpReduceSum(thread_sum);
    thread_sum_sq = warpReduceSum(thread_sum_sq);

    if (lane_id == 0) {
      scalar_t group_mean = thread_sum / group_elems;
      scalar_t group_var  = thread_sum_sq / group_elems - group_mean * group_mean;
      int out_index = n * num_groups + g;
      mean[out_index] = group_mean;
      var[out_index] = group_var;
    }
  }
}

// Group normalization forward kernel: applies normalization using precomputed mean and variance
// Grid-stride loop over all elements
template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int N,
    const int C,
    const int spatial,            // product of dimensions from index 2 onward
    const int channels_per_group, // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = N * C * spatial;

  for (; idx < total; idx += stride) {
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

// Host function to launch the optimized kernels with experimentally tuned block sizes
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

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
  auto var  = torch::empty({N, num_groups}, options);

  const int total_groups = N * num_groups;
  const dim3 blocks_stats(total_groups);

  // Use tuned block size for stats kernel
  const int threads_stats = BLOCK_SIZE_STATS;
  // Shared memory for inter-warp reduction: one entry per warp
  size_t shared_mem_size = (threads_stats / 32) * 2 * sizeof(float);

  const int total_elements = N * C * spatial;
  const int threads_norm = BLOCK_SIZE_NORM;
  int blocks_norm = (total_elements + threads_norm - 1) / threads_norm;

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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with optimal block sizes");
}
