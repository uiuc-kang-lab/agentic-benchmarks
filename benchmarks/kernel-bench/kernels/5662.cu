#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// Constant memory declaration for kernel config parameters
__constant__ struct PoolConfig {
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int input_height;
    int input_width;
} config;

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_maxpool(
    const scalar_t* __restrict__ input,
    int b, int c, int oh, int ow,
    int channels) {
  
  scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
  const int input_plane = config.input_height * config.input_width;
  const int offset = b * channels * input_plane + c * input_plane;

  // Use config from constant memory
  for (int kh = 0; kh < config.kernel_size; ++kh) {
    for (int kw = 0; kw < config.kernel_size; ++kw) {
      int ih = oh * config.stride - config.padding + kh * config.dilation;
      int iw = ow * config.stride - config.padding + kw * config.dilation;
      if (ih >= 0 && ih < config.input_height && iw >= 0 && iw < config.input_width) {
        int idx = offset + ih * config.input_width + iw;
        scalar_t val = input[idx];
        max_val = max(max_val, val);
      }
    }
  }
  return max_val;
}

template <typename scalar_t>
__global__ void maxpool2d_constant_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int output_height,
    const int output_width) {

  const int total = batch_size * channels * output_height * output_width;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (idx < total) {
    const int ow = idx % output_width;
    const int oh = (idx / output_width) % output_height;
    const int c = (idx / (output_width * output_height)) % channels;
    const int b = idx / (output_width * output_height * channels);

    output[idx] = compute_maxpool(input, b, c, oh, ow, channels);
    idx += blockDim.x * gridDim.x;
  }
}

torch::Tensor const_config_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {

  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
  
  const int input_height = input.size(2);
  const int input_width = input.size(3);
  
  // Copy config parameters to constant memory
  PoolConfig host_config = {
    kernel_size, stride, padding, dilation, input_height, input_width
  };
  cudaMemcpyToSymbol(config, &host_config, sizeof(PoolConfig));

  const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
  const int output_width = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

  auto output = torch::empty({input.size(0), input.size(1),
                        output_height, output_width}, input.options());

  const int total_elements = output.numel();
  const int threads = 256;
  const int blocks = (total_elements + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool_constant_kernel", ([&] {
    maxpool2d_constant_kernel<scalar_t><<<blocks, threads>>>(input.data_ptr<scalar_t>(),
                                                           output.data_ptr<scalar_t>(),
                                                           input.size(0),
                                                           input.size(1),
                                                           output_height,
                                                           output_width);
  }));

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &const_config_forward, "Max Pool 2D with constant config (CUDA)");
}
