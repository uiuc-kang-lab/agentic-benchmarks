#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Define optimal block size based on empirical testing
#define BLOCK_SIZE 256

// Optimized warp reduction with no divergent branches
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Compute statistics kernel with combined 1D and 2D grid strategies
// Each block computes the mean and variance for one (n, g) group
// Using 1D grid for simplicity and dynamic parallelism
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

  // Use 1D grid: blockIdx.x: (n, g) pair
  int idx = blockIdx.x;
  int n = idx / num_groups;
  int g = idx % num_groups;

  // Compute starting offset for the current group in sample n
  int group_offset = n * C * spatial + g * channels_per_group * spatial;
  int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;

  // Each thread processes multiple elements via striding over group_elems
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    scalar_t val = x[group_offset + i];
    sum += val;
    sum_sq += val * val;
  }

  // Warp-level reduction
  sum = warpReduceSum(sum);
  sum_sq = warpReduceSum(sum_sq);

  // Shared memory for inter-warp reduction
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + (blockDim.x / warpSize);

  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  if (lane_id == 0) {
    s_sum[warp_id] = sum;
    s_sum_sq[warp_id] = sum_sq;
  }
  __syncthreads();

  // Final reduction: first warp aggregates results from each warp
  if (warp_id == 0 && lane_id < ((blockDim.x + warpSize - 1) / warpSize)) {
    sum = s_sum[lane_id];
    sum_sq = s_sum_sq[lane_id];
    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    if (lane_id == 0) {
      scalar_t group_mean = sum / group_elems;
      scalar_t group_var = sum_sq / group_elems - group_mean * group_mean;
      int out_index = n * num_groups + g;
      mean[out_index] = group_mean;
      var[out_index] = group_var;
    }
  }
}

// Group Norm forward kernel using improved grid-stride loop
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

// Host function that launches the kernels
// Utilizes optimal block sizes and shared memory usage
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
  auto var = torch::empty({N, num_groups}, options);

  const int total_groups = N * num_groups;
  const dim3 blocks_stats(total_groups);
  const int threads_stats = BLOCK_SIZE;
  size_t shared_mem_size = (threads_stats / 32) * 2 * sizeof(float);

  const int total_elements = N * C * spatial;
  const int threads_norm = BLOCK_SIZE;
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
  m.def("forward", &group_norm_forward, "Optimized Group Normalization forward (CUDA)");
}
