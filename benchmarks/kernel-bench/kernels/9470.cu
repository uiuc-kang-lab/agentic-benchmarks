#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Inline clamp function to ensure indices are within bounds
// This simple version uses the ternary operator, which the compiler can predicated
__device__ inline int clamp_int(int x, int low, int high) {
    return x < low ? low : (x > high ? high : x);
}

// CUDA kernel for 2D transposed convolution with minimized warp divergence
// The kernel removes conditional branches in the inner loops by computing validity masks
// and clamping indices to safe values. Contributions from invalid positions are multiplied by 0.
__global__ void conv_transpose2d_forward_kernel_nodivergence(
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
  if (index >= total)
      return;

  // Decode thread index into (b, o, out_h, out_w)
  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  float out_val = bias[o];

  // Loop over input channels
  for (int c = 0; c < in_channels; ++c) {
    // For each kernel row
    for (int p = 0; p < kernel_size; ++p) {
      int tmp_h = h_out + padding - p * dilation;
      int h_div = tmp_h / stride;
      int h_mod = tmp_h - h_div * stride; // equivalent to tmp_h % stride in a branchless way
      // Compute validity for h: valid if remainder is 0 and h_div is within [0, in_height)
      int valid_h = (int)(h_mod == 0) * (int)(((unsigned)h_div) < ((unsigned)in_height));
      // Clamp h_div to a safe value to avoid out-of-bound memory accesses
      int safe_h = clamp_int(h_div, 0, in_height - 1);

      // For each kernel column
      for (int q = 0; q < kernel_size; ++q) {
        int tmp_w = w_out + padding - q * dilation;
        int w_div = tmp_w / stride;
        int w_mod = tmp_w - w_div * stride;
        int valid_w = (int)(w_mod == 0) * (int)(((unsigned)w_div) < ((unsigned)in_width));
        int valid = valid_h * valid_w;  // 1 if both conditions are met, else 0
        int safe_w = clamp_int(w_div, 0, in_width - 1);

        int input_idx = ((b * in_channels + c) * in_height + safe_h) * in_width + safe_w;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        // Multiply contribution by the validity mask to zero out invalid accesses
        out_val += valid * input[input_idx] * weight[weight_idx];
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_nodivergence(
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
  int threads = 1024;
  int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_forward_kernel_nodivergence<<<blocks, threads>>>(
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
  if(err != cudaSuccess) {
    printf("Error in conv_transpose2d_forward_kernel_nodivergence: %s\n", cudaGetErrorString(err));
  }
  
  return output;
}

// Wrapper function to handle possible None bias
torch::Tensor conv_transpose2d_forward_wrapper_nodivergence(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int dilation) {

  int out_channels = weight.size(1);
  torch::Tensor bias;
  if(bias_obj.is(pybind11::none())) {
    bias = torch::zeros({out_channels}, weight.options());
  } else {
    bias = bias_obj.cast<torch::Tensor>();
  }

  return conv_transpose2d_forward_cuda_nodivergence(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_nodivergence,
        "ConvTranspose2d forward with minimized warp divergence (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
