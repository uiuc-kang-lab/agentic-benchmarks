#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <algorithm>

// Declare constant memory for weight and bias (assuming max channels <= 1024 and type float)
__constant__ float c_weight_const[1024];
__constant__ float c_bias_const[1024];

// Kernel to compute per-group mean and variance for a chunk of data (processing local_N samples)
// x has shape: (local_N, C, spatial), mean and var have shape: (local_N, num_groups)
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,   // pointer to input chunk
    const int local_N,                  // number of samples in this chunk (batch dimension)
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,        // output: (local_N, num_groups)
    scalar_t* __restrict__ var) {         // output: (local_N, num_groups)

  // Each block processes one (n, g) pair, where n is relative to the chunk
  int idx = blockIdx.x;
  int n = idx / num_groups;  // within current chunk
  int g = idx % num_groups;

  int group_offset = n * C * spatial + g * channels_per_group * spatial;
  int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    int c = i / spatial;
    int j = i % spatial;
    scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;

  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scalar_t group_mean = s_sum[0] / group_elems;
    scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
    mean[n * num_groups + g] = group_mean;
    var[n * num_groups + g] = group_var;
  }
}

// Kernel to apply group normalization for a chunk of data
// x and y have shape: (local_N, C, spatial); mean and var have shape: (local_N, num_groups)
template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,    // input chunk pointer
    const scalar_t* __restrict__ mean,   // computed mean for this chunk
    const scalar_t* __restrict__ var,    // computed var for this chunk
    const int local_N,                   // number of samples in current chunk
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {          // output chunk pointer

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = local_N * C * spatial;
  if (idx >= total) return;

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
  // Use constant memory for weight and bias
  y[idx] = (x_val - m) * inv_std * c_weight_const[c] + c_bias_const[c];
}

// Host function using pipelining to overlap memory transfers with computation
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

  // x expected shape: (N, C, *), where * are the spatial dimensions
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

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward", ([&] {
    using scalar_t = float;  // Assumes float precision

    // Asynchronously copy weight and bias to constant memory using a dedicated copy stream
    cudaStream_t copy_stream;
    cudaStreamCreate(&copy_stream);
    cudaMemcpyToSymbolAsync(c_weight_const, weight.data_ptr<scalar_t>(), C * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice, copy_stream);
    cudaMemcpyToSymbolAsync(c_bias_const, bias.data_ptr<scalar_t>(), C * sizeof(scalar_t), 0, cudaMemcpyDeviceToDevice, copy_stream);
    cudaEvent_t copy_done_event;
    cudaEventCreate(&copy_done_event);
    cudaEventRecord(copy_done_event, copy_stream);
    cudaStreamDestroy(copy_stream);

    // Create two CUDA streams for pipelining
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Determine chunk size along the batch dimension (N) for pipelining
    int chunk_size = (N > 16) ? (N / 2) : N;  // Use 2 chunks if possible

    for (int n_offset = 0; n_offset < N; n_offset += chunk_size) {
      int current_chunk = std::min(chunk_size, N - n_offset);
      // Compute pointer offsets for the current chunk
      const scalar_t* x_ptr = x.data_ptr<scalar_t>() + n_offset * C * spatial;
      scalar_t* y_ptr = y.data_ptr<scalar_t>() + n_offset * C * spatial;
      scalar_t* mean_ptr = mean.data_ptr<scalar_t>() + n_offset * num_groups;
      scalar_t* var_ptr = var.data_ptr<scalar_t>() + n_offset * num_groups;
      
      int stream_id = ((n_offset / chunk_size) % 2);
      // Wait for the constant memory copy to complete in the current stream
      cudaStreamWaitEvent(streams[stream_id], copy_done_event, 0);

      // Launch compute_stats_kernel for this chunk
      int total_groups_chunk = current_chunk * num_groups;
      int threads_stats = 256;
      dim3 blocks_stats(total_groups_chunk);
      size_t shared_mem_size = threads_stats * 2 * sizeof(scalar_t);

      compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, streams[stream_id]>>>(
          x_ptr,
          current_chunk,
          C,
          spatial,
          channels_per_group,
          num_groups,
          mean_ptr,
          var_ptr);

      // Launch group_norm_forward_kernel for this chunk
      int total_elements = current_chunk * C * spatial;
      int threads_norm = 256;
      dim3 blocks_norm((total_elements + threads_norm - 1) / threads_norm);

      group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, 0, streams[stream_id]>>>(
          x_ptr,
          mean_ptr,
          var_ptr,
          current_chunk,
          C,
          spatial,
          channels_per_group,
          num_groups,
          static_cast<scalar_t>(eps),
          y_ptr);
    }

    // Synchronize the streams before returning
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaEventDestroy(copy_done_event);
  }));

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_norm_forward, "Pipelined Group Normalization forward with stream overlapping (CUDA)");
}
