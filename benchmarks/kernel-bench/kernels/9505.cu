#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// CUDA kernel for 2D transposed convolution (conv_transpose2d) forward with shared memory optimization.
__global__ void conv_transpose2d_forward_kernel_shared(
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

  // Decode index into (b, o, out_h, out_w)
  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  float out_val = bias[o]; // start with bias

  for (int c = 0; c < in_channels; ++c) {
    // Load weights into shared memory
    if (threadIdx.x < kernel_size * kernel_size) {
      int p = threadIdx.x / kernel_size;
      int q = threadIdx.x % kernel_size;
      shared_weight[threadIdx.x] = weight[((c * out_channels + o) * kernel_size + p) * kernel_size + q];
    }
    __syncthreads();

    for (int p = 0; p < kernel_size; ++p) {
      int h_unscaled = h_out + padding - p * dilation;
      if (h_unscaled % stride != 0)
        continue;
      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height)
        continue;
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = w_out + padding - q * dilation;
        if (w_unscaled % stride != 0)
          continue;
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width)
          continue;
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        out_val += input[input_idx] * shared_weight[p * kernel_size + q];
      }
    }
    __syncthreads();
  }

  int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}

torch::Tensor conv_transpose2d_forward_cuda_shared(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
  
  // Get input dimensions.
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);
  
  // Weight tensor: [in_channels, out_channels, kernel_size, kernel_size]
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);  // assume square kernel
  
  // Calculate output dimensions.
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  
  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
  
  int total_threads = batch_size * out_channels * out_height * out_width;
  int threads = 1024;
  int blocks = (total_threads + threads - 1) / threads;
  int shared_mem_size = kernel_size * kernel_size * sizeof(float);
  
  conv_transpose2d_forward_kernel_shared<<<blocks, threads, shared_mem_size>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_shared: %s\n", cudaGetErrorString(err));
  }
  
  return output;
}

// Wrapper function to handle the possibility that the bias is None.
// If bias is None, we create a zero bias tensor of shape [out_channels].
torch::Tensor conv_transpose2d_forward_wrapper_shared(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,  // using py::object to accept None
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
  
  return conv_transpose2d_forward_cuda_shared(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_shared,
        "ConvTranspose2d forward with shared memory (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}