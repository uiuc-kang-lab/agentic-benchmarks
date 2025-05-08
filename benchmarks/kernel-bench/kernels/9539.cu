#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// CUDA kernel for 2D transposed convolution (conv_transpose2d) forward with memory coalescing optimization.
__global__ void conv_transpose2d_forward_kernel_optimized(
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

  extern __shared__ float shared_weight[];

  int w_out = blockIdx.x * blockDim.x + threadIdx.x;
  int h_out = blockIdx.y * blockDim.y + threadIdx.y;
  int o = blockIdx.z;

  if (w_out >= out_width || h_out >= out_height || o >= out_channels)
    return;

  float out_val = bias[o];

  for (int c = 0; c < in_channels; ++c) {
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
        int input_idx = ((blockIdx.z * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;

        if (threadIdx.y == 0 && threadIdx.x < kernel_size * kernel_size) {
          shared_weight[threadIdx.x] = weight[weight_idx];
        }
        __syncthreads();

        out_val += input[input_idx] * shared_weight[(p * kernel_size) + q];
      }
    }
  }

  int output_idx = ((blockIdx.z * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}


torch::Tensor conv_transpose2d_forward_cuda_optimized(
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
  
  dim3 threads(16, 16);
  dim3 blocks((out_width + threads.x - 1) / threads.x, (out_height + threads.y - 1) / threads.y, out_channels);
  size_t shared_memory_size = kernel_size * kernel_size * sizeof(float);
  
  conv_transpose2d_forward_kernel_optimized<<<blocks, threads, shared_memory_size>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_optimized: %s\n", cudaGetErrorString(err));
  }
  
  return output;
}

// Wrapper function to handle the possibility that the bias is None.
// If bias is None, we create a zero bias tensor of shape [out_channels].
torch::Tensor conv_transpose2d_forward_wrapper_optimized(
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
  
  return conv_transpose2d_forward_cuda_optimized(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_optimized,
        "ConvTranspose2d forward optimized (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}