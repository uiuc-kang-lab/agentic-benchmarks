#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Define maximum channels for constant memory usage
#define MAX_CHANNELS 2048

// Declare constant memory arrays for weight and bias for float and double types
__constant__ float const_weight_float[MAX_CHANNELS];
__constant__ float const_bias_float[MAX_CHANNELS];

__constant__ double const_weight_double[MAX_CHANNELS];
__constant__ double const_bias_double[MAX_CHANNELS];

// Templated device helper functions to get weight and bias from constant memory
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_weight(int idx);

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_bias(int idx);

// Specialization for float
template <>
__device__ __forceinline__ float get_weight<float>(int idx) {
    return const_weight_float[idx];
}

template <>
__device__ __forceinline__ float get_bias<float>(int idx) {
    return const_bias_float[idx];
}

// Specialization for double
template <>
__device__ __forceinline__ double get_weight<double>(int idx) {
    return const_weight_double[idx];
}

template <>
__device__ __forceinline__ double get_bias<double>(int idx) {
    return const_bias_double[idx];
}

// -----------------------------------------------------------------------------
// Optimized Compute Statistics Kernel
// Combines the synchronous reduction techniques and the use of shared memory

template <typename scalar_t>
__global__ void compute_stats_kernel_combined(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

  const int idx = blockIdx.x;
  const int n = idx / num_groups;
  const int g = idx % num_groups;
  const int group_offset = n * C * spatial + g * channels_per_group * spatial;
  const int group_elems = channels_per_group * spatial;

  scalar_t sum = 0;
  scalar_t sum_sq = 0;
  for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
    const int c = i / spatial;
    const int j = i % spatial;
    const scalar_t val = x[group_offset + c * spatial + j];
    sum += val;
    sum_sq += val * val;
  }

  extern __shared__ char smem[];
  scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
  scalar_t* s_sum_sq = s_sum + blockDim.x;
  s_sum[threadIdx.x] = sum;
  s_sum_sq[threadIdx.x] = sum_sq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x < 32) {
    volatile scalar_t local_sum = s_sum[threadIdx.x];
    volatile scalar_t local_sum_sq = s_sum_sq[threadIdx.x];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
      local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    if (threadIdx.x == 0) {
      const scalar_t group_mean = local_sum / group_elems;
      const scalar_t group_var = local_sum_sq / group_elems - group_mean * group_mean;
      const int out_index = n * num_groups + g;
      mean[out_index] = group_mean;
      var[out_index] = group_var;
    }
  }
}

// -----------------------------------------------------------------------------
// Group normalization forward kernel using constant memory for weight and bias

// Unchanged as it's already efficient

// -----------------------------------------------------------------------------
// Host function: Group normalization forward pass

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
  const int group_elems = channels_per_group * spatial;
  const int threads_stats = (group_elems < 256 ? group_elems : 256);
  const dim3 blocks_stats(total_groups);

  const int total_elements = N * C * spatial;
  const int threads_norm = 256;
  const dim3 blocks_norm((total_elements + threads_norm - 1) / threads_norm);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
    if (std::is_same<scalar_t, float>::value) {
      cudaMemcpyToSymbol(const_weight_float, weight.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
      cudaMemcpyToSymbol(const_bias_float, bias.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyToSymbol(const_weight_double, weight.data_ptr<double>(), C * sizeof(double), 0, cudaMemcpyDeviceToDevice);
      cudaMemcpyToSymbol(const_bias_double, bias.data_ptr<double>(), C * sizeof(double), 0, cudaMemcpyDeviceToDevice);
    }
    
    const size_t shared_mem_size = threads_stats * 2 * sizeof(scalar_t);

    compute_stats_kernel_combined<scalar_t><<<blocks_stats, threads_stats, shared_mem_size, stream>>>(
        x.data_ptr<scalar_t>(),
        N,
        C,
        spatial,
        channels_per_group,
        num_groups,
        mean.data_ptr<scalar_t>(),
        var.data_ptr<scalar_t>());

    group_norm_forward<scalar_t><<<blocks_norm, threads_norm, 0, stream>>>(
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
  m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA)");
}
