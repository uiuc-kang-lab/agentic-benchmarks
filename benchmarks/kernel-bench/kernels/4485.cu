#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Use float4 for vectorized loads (ensuring 16-byte alignment)
typedef float4 float4_t;

// Optimized warp reduction using shuffle
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Kernel to compute per-group mean and variance with aligned, coalesced accesses
// Each block processes one (n, g) pair
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,            // product of dimensions from index 2 onward
    const int channels_per_group, // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,  // shape: (N, num_groups)
    scalar_t* __restrict__ var) { // shape: (N, num_groups)

  // Each block is assigned one (n, g) pair
  const int idx = blockIdx.x; 
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  
  // Calculate starting offset for this group
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  // Use vectorized loads if possible - we assume the pointer is aligned
  const int vec_size = 4; // float4
  const int num_vectors = group_elems / vec_size;
  const int remaining = group_elems % vec_size;

  scalar_t local_sum = 0;
  scalar_t local_sum_sq = 0;

  // Cast pointer for vectorized (aligned) loads
  const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);

  // Each thread processes multiple vectors in a grid-stride loop
  for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    float4_t v = __ldg(&x_vec[i]);
    local_sum    += v.x + v.y + v.z + v.w;
    local_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }

  // Handle remaining elements
  if (threadIdx.x < remaining) {
    int rem_idx = num_vectors * vec_size + threadIdx.x;
    scalar_t val = __ldg(x + group_offset + rem_idx);
    local_sum    += val;
    local_sum_sq += val * val;
  }

  // Intra-warp reduction using shuffle
  local_sum = warpReduceSum(local_sum);
  local_sum_sq = warpReduceSum(local_sum_sq);

  // Allocate shared memory for inter-warp reduction
  __shared__ scalar_t sh_sum[32];  // one entry per warp
  __shared__ scalar_t sh_sum_sq[32];

  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  if (lane == 0) {
    sh_sum[warp_id] = local_sum;
    sh_sum_sq[warp_id] = local_sum_sq;
  }
  __syncthreads();

  // Final reduction by first warp
  if (warp_id == 0) {
    local_sum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? sh_sum[lane] : 0;
    local_sum_sq = (lane < (blockDim.x + warpSize - 1) / warpSize) ? sh_sum_sq[lane] : 0;
    local_sum = warpReduceSum(local_sum);
    local_sum_sq = warpReduceSum(local_sum_sq);
    if (lane == 0) {
      scalar_t group_mean = local_sum / group_elems;
      scalar_t group_var  = local_sum_sq / group_elems - group_mean * group_mean;
      const int out_index = n * num_groups + g;
      mean[out_index] = group_mean;
      var[out_index] = group_var;
    }
  }
}

// Kernel to perform group normalization with coalesced reads/writes
// Threads are mapped so that consecutive threads access consecutive memory locations
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

  const int total = N * C * spatial;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  // Each thread processes consecutive elements
  for (int i = idx; i < total; i += stride) {
    // Decode index into (n, c, spatial_pos)
    int j = i % spatial;
    int temp = i / spatial;
    int c = temp % C;
    int n = temp / C;
    
    // Determine the group for channel c
    int g = c / channels_per_group;
    int stats_idx = n * num_groups + g;

    // Use __ldg() for read-only accesses to improve caching
    scalar_t m = __ldg(&mean[stats_idx]);
    scalar_t v = __ldg(&var[stats_idx]);
    scalar_t inv_std = rsqrt(v + eps);
    scalar_t x_val = __ldg(&x[i]);
    scalar_t w = __ldg(&weight[c]);
    scalar_t b = __ldg(&bias[c]);

    // Write result - consecutive threads write consecutive locations
    y[i] = (x_val - m) * inv_std * w + b;
  }
}

// Host function that launches the kernels
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

  // Launch compute_stats_kernel: one block per (n, group) pair
  const int total_groups = N * num_groups;
  const dim3 blocks_stats(total_groups);
  const int threads_stats = 256;

  // Shared memory for reduction (each warp contributes one float)
  size_t shared_mem = (threads_stats / 32) * 2 * sizeof(float);

  // Launch kernel on the current CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    // Launch group normalization forward kernel
    const int total_elements = N * C * spatial;
    const int threads_norm = 256;
    const int blocks_norm = (total_elements + threads_norm - 1) / threads_norm;
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with aligned & coalesced accesses");
}
