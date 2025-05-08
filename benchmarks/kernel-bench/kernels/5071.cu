#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_forward_kernel_streamed(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int instances_per_stream) {

  int instance_within_stream = blockIdx.x % instances_per_stream;
  int stream_offset = (blockIdx.x / instances_per_stream) * instances_per_stream * normalized_size;
  int instance_idx = instance_within_stream;
  int tid = threadIdx.x;

  const scalar_t* in_ptr = input + stream_offset + instance_idx * normalized_size;
  scalar_t* out_ptr = output + stream_offset + instance_idx * normalized_size;

  using accscalar_t = at::acc_type<scalar_t, true>;

  extern __shared__ char smem[];
  accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
  accscalar_t* s_sum_sq = s_sum + blockDim.x;

  accscalar_t local_sum = 0;
  accscalar_t local_sum_sq = 0;

  #pragma unroll 4
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    scalar_t val = __ldg(&in_ptr[i]);
    accscalar_t acc_val = static_cast<accscalar_t>(val);
    local_sum += acc_val;
    local_sum_sq += acc_val * acc_val;
  }

  s_sum[tid] = local_sum;
  s_sum_sq[tid] = local_sum_sq;
  __syncthreads();

  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
      s_sum_sq[tid] += s_sum_sq[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    accscalar_t final_sum = 0;
    accscalar_t final_sum_sq = 0;
    for (int i = 0; i < blockDim.x/warpSize; ++i) {
      final_sum += s_sum[i * warpSize];
      final_sum_sq += s_sum_sq[i * warpSize];
    }
    s_sum[0] = final_sum;
    s_sum_sq[0] = final_sum_sq;
  }
  __syncthreads();

  __shared__ accscalar_t mean, inv_std;
  if (tid == 0) {
    mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = rsqrt(var + static_cast<accscalar_t>(eps));
  }
  __syncthreads();

  #pragma unroll 4
  for (int i = tid; i < normalized_size; i += blockDim.x) {
    scalar_t in_val = __ldg(&in_ptr[i]);
    scalar_t w = __ldg(&weight[i]);
    scalar_t b = __ldg(&bias[i]);
    
    accscalar_t normalized = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
    out_ptr[i] = static_cast<scalar_t>(normalized * static_cast<accscalar_t>(w) + 
                                      static_cast<accscalar_t>(b));
  }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
  auto output = torch::empty_like(x);
  
  const int normalized_size = weight.numel();
  const int outer_size = x.numel() / normalized_size;
  
  const int num_streams = 4;
  const int instances_per_stream = (outer_size + num_streams - 1) / num_streams;
  
  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  const int threads = 256;
  const int blocks = outer_size;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    const int shared_size = threads * 2 * sizeof(accscalar_t);

    for (int i = 0; i < num_streams; i++) {
      const int stream_start = i * instances_per_stream;
      const int stream_size = std::min(instances_per_stream, outer_size - stream_start);
      
      if (stream_size <= 0) continue;

      const scalar_t* input_ptr = x.data_ptr<scalar_t>() + stream_start * normalized_size;
      scalar_t* output_ptr = output.data_ptr<scalar_t>() + stream_start * normalized_size;

      layernorm_forward_kernel_streamed<scalar_t><<<blocks, threads, shared_size, streams[i]>>>(
          input_ptr,
          weight.data_ptr<scalar_t>(),
          bias.data_ptr<scalar_t>(),
          static_cast<float>(eps),
          output_ptr,
          normalized_size,
          instances_per_stream);
    }
  }));

  for (int i = 0; i < num_streams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
        py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}