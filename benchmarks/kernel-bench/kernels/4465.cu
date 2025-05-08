#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Declare constant memory for weight and bias (assuming max channels <= 1024)
__constant__ float c_weight[1024];
__constant__ float c_bias[1024];

// Kernel to compute per-group mean and variance.
// Each block processes one (n, g) group.
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

  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  
  // Starting offset for group g of sample n
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  // Each thread processes several elements via striding
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    const int c = i / spatial;
    const int j = i % spatial;
    const scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  // Shared memory for reduction
  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  // Parallel reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Thread 0 writes the result
  if (threadIdx.x == 0) {
    const scalar_t group_mean = s_sum[0] / group_elems;
    const scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    mean[n * num_groups + g] = group_mean;
    var[n * num_groups + g] = group_var;
  }
}

// Kernel to apply group normalization using computed mean and variance.
// Each thread processes one element from the input tensor.
template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const int N,
    const int C,
    const int spatial,             // product of dimensions from index 2 onward
    const int channels_per_group,  // C / num_groups
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = N * C * spatial;
  if (index >= total)
    return;

  // Decode flattened index into (n, c, j) coordinates.
  const int j = index % spatial;
  const int temp = index / spatial;
  const int c = temp % C;
  const int n = temp / C;

  // Determine which group this channel belongs to
  const int g = c / channels_per_group;
  const int stats_index = n * num_groups + g;
  const scalar_t m = mean[stats_index];
  const scalar_t v = var[stats_index];
  const scalar_t inv_std = rsqrt(v + eps);
  const scalar_t x_val = x[index];
  
  // Use constant memory for weight and bias
  y[index] = (x_val - m) * inv_std * c_weight[c] + c_bias[c];
}

// Host function to launch the Group Normalization kernels
// Implements overlap of constant memory transfers with computation using CUDA streams
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
  const int channels_per_group = C / num_groups;

  // Prepare output tensor and temporary buffers for means and variances
  auto y = torch::empty_like(x);
  auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
  auto mean = torch::empty({N, num_groups}, options);
  auto var = torch::empty({N, num_groups}, options);

  // Create a separate CUDA stream for asynchronous copying of constant values
  cudaStream_t copy_stream;
  cudaStreamCreate(&copy_stream);

  // Asynchronously copy weight and bias to constant memory on the copy stream
  // Assuming weight and bias are already on the device
  cudaMemcpyToSymbolAsync(c_weight, weight.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice, copy_stream);
  cudaMemcpyToSymbolAsync(c_bias, bias.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice, copy_stream);

  // Record an event on the copy stream to signal completion of the copies
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  cudaEventRecord(copy_done, copy_stream);

  // Obtain the default compute stream and wait for the copy to complete
  cudaStream_t compute_stream = c10::cuda::getCurrentCUDAStream();
  cudaStreamWaitEvent(compute_stream, copy_done, 0);

  // Launch the compute_stats_kernel
  const int total_groups = N * num_groups;
  const int group_elems = channels_per_group * spatial;
  const int threads_stats = (group_elems < 256 ? group_elems : 256);
  const dim3 blocks_stats(total_groups);
  const size_t shared_mem_size = threads_stats * 2 * sizeof(float);

  // Launch the group norm forward kernel
  const int total_elements = N * C * spatial;
  const int threads_norm = 256;
  const dim3 blocks_norm((total_elements + threads_norm - 1) / threads_norm);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, compute_stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, 0, compute_stream>>>(
        x.data_ptr<scalar_t>(),
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        static_cast<scalar_t>(eps),
        y.data_ptr<scalar_t>());
  }));

  // Clean up the copy stream and event
  cudaEventDestroy(copy_done);
  cudaStreamDestroy(copy_stream);

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward, "Group Normalization forward with stream overlap (CUDA)");
}
