#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <pybind11/pybind11.h>

// Optimized CUDA kernel for 2D transposed convolution that computes the valid iteration bounds
// for the kernel indices (p and q) based on the output coordinates. This reduces unnecessary
// loop iterations and branch overhead. Note that we use the gather approach so that no atomic
// operations are required (using atomics only where necessary).

__global__ void conv_transpose2d_forward_kernel_opt_bounds(
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

  // Decode index into (b, o, out_h, out_w)
  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  float out_val = bias[o];

  // Compute adjusted output coordinates
  int temp_h = h_out + padding;
  int temp_w = w_out + padding;

  // Compute valid range for kernel index p
  float p_min_f = ceilf((temp_h - (in_height - 1) * (float)stride) / (float)dilation);
  int p_min = (int)p_min_f;
  if (p_min < 0) p_min = 0;
  int p_max_calc = (int)floorf(temp_h / (float)dilation);
  int p_max = p_max_calc;
  if (p_max > kernel_size - 1) p_max = kernel_size - 1;

  // Compute valid range for kernel index q
  float q_min_f = ceilf((temp_w - (in_width - 1) * (float)stride) / (float)dilation);
  int q_min = (int)q_min_f;
  if (q_min < 0) q_min = 0;
  int q_max_calc = (int)floorf(temp_w / (float)dilation);
  int q_max = q_max_calc;
  if (q_max > kernel_size - 1) q_max = kernel_size - 1;

  // Loop over input channels and only valid (p,q) that contribute to output element
  for (int c = 0; c < in_channels; ++c) {
    for (int p = p_min; p <= p_max; ++p) {
      int h_temp = temp_h - p * dilation;
      if (h_temp % stride != 0) continue;
      int h_in = h_temp / stride;
      if (h_in < 0 || h_in >= in_height) continue;
      for (int q = q_min; q <= q_max; ++q) {
        int w_temp = temp_w - q * dilation;
        if (w_temp % stride != 0) continue;
        int w_in = w_temp / stride;
        if (w_in < 0 || w_in >= in_width) continue;
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        out_val += input[input_idx] * weight[weight_idx];
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}

// CUDA forward function
torch::Tensor conv_transpose2d_forward_cuda_opt_bounds(
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
  int kernel_size = weight.size(2);  // assume square kernel

  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  int total_threads = batch_size * out_channels * out_height * out_width;
  int threads = 512;
  int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_forward_kernel_opt_bounds<<<blocks, threads>>>(
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

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in conv_transpose2d_forward_kernel_opt_bounds: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper to handle the case when bias is None
torch::Tensor conv_transpose2d_forward_wrapper_opt_bounds(
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

  return conv_transpose2d_forward_cuda_opt_bounds(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_opt_bounds,
        "Optimized ConvTranspose2d forward (CUDA) with computed valid bounds for kernel iteration",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
