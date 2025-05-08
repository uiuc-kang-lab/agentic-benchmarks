#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

__global__ void conv_transpose2d_optimized_kernel(
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

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_channels * out_height * out_width;
  if (index >= total) return;

  // Calculate output coordinates
  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  float val = __ldg(&bias[o]);

  // Find valid input positions that contribute to this output
  for (int h_in = 0; h_in < in_height; ++h_in) {
    int h_scaled = h_in * stride;
    int p_numerator = h_out + padding - h_scaled;
    
    if (p_numerator < 0 || p_numerator % dilation != 0) continue;
    int p = p_numerator / dilation;
    if (p < 0 || p >= kernel_size) continue;

    for (int w_in = 0; w_in < in_width; ++w_in) {
      int w_scaled = w_in * stride;
      int q_numerator = w_out + padding - w_scaled;
      
      if (q_numerator < 0 || q_numerator % dilation != 0) continue;
      int q = q_numerator / dilation;
      if (q < 0 || q >= kernel_size) continue;

      for (int c = 0; c < in_channels; ++c) {
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        
        val += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = val;
}

torch::Tensor conv_transpose2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
  
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);
  int out_channels = weight.size(0);
  int kernel_size = weight.size(2);

  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  int total = batch_size * out_channels * out_height * out_width;
  int threads = 256;  // Better occupancy for compute-bound kernels
  int blocks = (total + threads - 1) / threads;

  conv_transpose2d_optimized_kernel<<<blocks, threads>>>(
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
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_cuda_forward,
        "Optimized ConvTranspose2D forward",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
