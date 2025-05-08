__device__ __forceinline__ float compute_conv_transpose_at_pixel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size,
    int h_out, int w_out, int stride, int padding, int dilation, int o) {
    
  float sum = 0.0f;
  #pragma unroll 4
  for (int c = 0; c < in_channels; ++c) {
    for (int p = 0; p < kernel_size; ++p) {
      int h_unscaled = h_out + padding - p * dilation;
      if (h_unscaled % stride != 0) continue;
      
      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height) continue;
      
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = w_out + padding - q * dilation;
        if (w_unscaled % stride != 0) continue;
        
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width) continue;
        
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        sum = __fmaf_rn(input[input_idx], weight[weight_idx], sum);
      }
    }
  }
  return sum;
}

__global__ void conv_transpose2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation) {

  int w_out = blockIdx.x * blockDim.x + threadIdx.x;
  int h_out = blockIdx.y * blockDim.y + threadIdx.y;
  int batch_channel_idx = blockIdx.z;
  
  int b = batch_channel_idx / out_channels;
  int o = batch_channel_idx % out_channels;

  if (b >= batch_size || o >= out_channels || h_out >= out_height || w_out >= out_width)
    return;

  float out_val = bias[o] + compute_conv_transpose_at_pixel(
    input, weight, b, in_channels, in_height, in_width,
    out_channels, kernel_size, h_out, w_out,
    stride, padding, dilation, o);

  int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Host wrapper to launch the CUDA kernel
void conv_transpose2d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int stride,
    int padding,
    int dilation) {
  const auto batch_size = input.size(0);
  const auto in_channels = input.size(1);
  const auto in_height = input.size(2);
  const auto in_width = input.size(3);
  const auto out_channels = weight.size(1);
  const auto kernel_size = weight.size(2);  // assuming square kernel
  const auto out_height = output.size(2);
  const auto out_width = output.size(3);

  dim3 block(16, 16);
  dim3 grid((out_width + block.x - 1) / block.x,
            (out_height + block.y - 1) / block.y,
            batch_size * out_channels);

  conv_transpose2d_forward_kernel<<<grid, block>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      output.data_ptr<float>(),
      batch_size,
      in_channels,
      out_channels,
      in_height,
      in_width,
      kernel_size,
      out_height,
      out_width,
      stride,
      padding,
      dilation);

  cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_transpose2d_forward", &conv_transpose2d_forward, "Conv Transpose 2D forward (CUDA)");
}
