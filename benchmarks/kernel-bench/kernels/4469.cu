#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Kernel to compute per-group mean and variance using a stride loop
// Each block handles one (n, g) group from the input tensor
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    scalar_t* __restrict__ mean,   // shape: (N, num_groups)
    scalar_t* __restrict__ var) {  // shape: (N, num_groups)

  // Each block corresponds to one (n, g) pair
  int idx = blockIdx.x; 
  int n = idx / num_groups;
  int g = idx % num_groups;
  
  // Starting offset for group g of sample n
  int group_offset = n * C * spatial + g * channels_per_group * spatial;
  int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  // Use a stride loop so that threads cover all elements, even if group_elems > blockDim.x
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // Shared memory for reduction; allocate space for sum and sum of squares
  extern __shared__ char shared_mem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(shared_mem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;

  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  // Reduce within the block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Thread 0 writes the final result
  if (threadIdx.x == 0) {
    scalar_t group_mean = s_sum[0] / group_elems;
    scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    int out_idx = n * num_groups + g;
    mean[out_idx] = group_mean;
    var[out_idx] = group_var;
  }
}

// Kernel to apply Group Normalization using a grid-stride loop for boundary handling
// Each thread processes multiple elements if necessary
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

  int total = N * C * spatial;
  // Grid-stride loop to ensure all elements are processed
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += blockDim.x * gridDim.x) {
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

// Host function to launch the CUDA kernels for Group Normalization
// Implements explicit stride loops for both the reduction and normalization kernels
torch::Tensor group_norm_forward(
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
  int channels_per_group = C / num_groups;

  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var  = torch::empty({N, num_groups}, options);

  int total_groups = N * num_groups;
  int threads_stats = 256;  // Use 256 threads for reduction
  dim3 blocks_stats(total_groups);

  int total_elements = N * C * spatial;
  int threads_norm = 256;
  dim3 blocks_norm((total_elements + threads_norm - 1) / threads_norm);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    size_t shared_mem_size = threads_stats * 2 * sizeof(scalar_t);
    
    // Launch the compute_stats_kernel with a stride loop for correct boundary handling
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());
    
    // Launch the normalization kernel using a grid-stride loop
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with stride loops");
}
