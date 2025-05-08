#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Declare constant memory for weights (48KB limit on most GPUs)
__constant__ float const_weight[16384]; // 64KB of constant memory

__global__ void conv_transpose2d_forward_kernel_constant(
    const float* __restrict__ input,
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

  int out_w = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = blockIdx.y * blockDim.y + threadIdx.y;
  int bo_idx = blockIdx.z;
  int o = bo_idx % out_channels;
  int b = bo_idx / out_channels;

  if (out_w >= out_width || out_h >= out_height)
    return;

  float out_val = bias[o];

  #pragma unroll 4
  for (int c = 0; c < in_channels; ++c) {
    #pragma unroll
    for (int p = 0; p < kernel_size; ++p) {
      int h_unscaled = out_h + padding - p * dilation;
      if (h_unscaled % stride != 0)
        continue;
      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height)
        continue;

      #pragma unroll
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = out_w + padding - q * dilation;
        if (w_unscaled % stride != 0)
          continue;
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width)
          continue;

        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        out_val += input[input_idx] * const_weight[weight_idx];
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
  output[output_idx] = out_val;
}

torch::Tensor conv_transpose2d_forward_cuda_constant(
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
  
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);
  
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  
  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Copy weights to constant memory
  size_t weight_size = weight.numel() * sizeof(float);
  cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), weight_size);

  // Configure kernel launch parameters
  dim3 block(32, 8);
  dim3 grid((out_width + block.x - 1) / block.x,
            (out_height + block.y - 1) / block.y,
            batch_size * out_channels);

  conv_transpose2d_forward_kernel_constant<<<grid, block>>>(
      input.data_ptr<float>(),
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
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in conv_transpose2d_forward_kernel_constant: %s\n", cudaGetErrorString(err));
  }
  
  return output;
}

torch::Tensor conv_transpose2d_forward_wrapper_constant(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int dilation) {
  
  int out_channels = weight.size(1);
  torch::Tensor bias;
  if (bias_obj.is(pybind11::none())) {
    bias = torch::zeros({out_channels}, weight.options());
  } else {
    bias = bias_obj.cast<torch::Tensor>();
  }
  
  return conv_transpose2d_forward_cuda_constant(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_constant,
        "ConvTranspose2d forward with constant memory (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}