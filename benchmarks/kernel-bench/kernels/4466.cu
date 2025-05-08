#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Tuning parameters: adjust block sizes for optimal performance on NVIDIA H100
#define BLOCK_SIZE_STATS 128
#define BLOCK_SIZE_NORM 512

// Kernel to compute per-group mean and variance using shared memory reduction.
// Each block handles one (n, g) group.
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // output shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // output shape: (N, num_groups)

  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  
  // Starting offset for group g of sample n
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  // Each thread processes multiple elements via striding
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // Allocate shared memory for reduction
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + BLOCK_SIZE_STATS;

  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // First thread writes the mean and variance
  if (threadIdx.x == 0) {
    scalar_t m = s_sum[0] / group_elems;
    scalar_t v = s_sum_sq[0] / group_elems - m * m;
    int out_index = n * num_groups + g;
    mean[out_index] = m;
    var[out_index] = v;
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

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * C * spatial;
  if (index >= total) return;

  // Decode flattened index into (n, c, spatial_idx)
  int j = index % spatial;
  int temp = index / spatial;
  int c = temp % C;
  int n = temp / C;

  int g = c / channels_per_group;
  int stats_index = n * num_groups + g;
  scalar_t m = mean[stats_index];
  scalar_t v = var[stats_index];
  scalar_t inv_std = rsqrt(v + eps);
  scalar_t val = x[index];
  y[index] = (val - m) * inv_std * weight[c] + bias[c];
}

// Host function to launch the CUDA kernels for Group Normalization
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
  // x is expected to have shape (N, C, *)
  const int N = x.size(0);
  const int C = x.size(1);
  int spatial = 1;
  for (int i = 2; i < x.dim(); i++) {
    spatial *= x.size(i);
  }
  const int channels_per_group = C / num_groups;

  // Create output tensor similar to x
  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Setup grid and block dimensions using tuned block sizes
  const int total_groups = N * num_groups;
  const int group_elems = channels_per_group * spatial;
  const dim3 blocks_stats(total_groups);
  const int threads_stats = (group_elems < BLOCK_SIZE_STATS ? group_elems : BLOCK_SIZE_STATS);
  size_t shared_mem_size = threads_stats * 2 * (x.scalar_type() == at::kDouble ? sizeof(double) : sizeof(float));

  const int total_elements = N * C * spatial;
  const dim3 blocks_norm((total_elements + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM);
  const int threads_norm = BLOCK_SIZE_NORM;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    // Launch statistics kernel
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, stream>>>(
      x.data_ptr<scalar_t>(),
      N,
      C,
      spatial,
      channels_per_group,
      num_groups,
      mean.data_ptr<scalar_t>(),
      var.data_ptr<scalar_t>()
    );
    
    // Launch normalization kernel
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
      y.data_ptr<scalar_t>()
    );
  }));

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with tuned block sizes");
}
