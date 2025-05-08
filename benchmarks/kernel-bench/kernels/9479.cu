#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Kernel using a 3D grid where grid.x = batch, grid.y = output channel, and grid.z covers spatial tiles in the output
// Each thread in a 1D block computes one output element from the flattened spatial dimensions
__global__ void conv_transpose2d_forward_kernel_spatial_tile(
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
    int dilation,
    int spatial_size) {

  // Use grid dimensions: grid.x = batch index, grid.y = output channel
  int b = blockIdx.x;
  int o = blockIdx.y;

  // grid.z covers tiles of the flattened spatial dimensions
  int tile_offset = blockIdx.z * blockDim.x;
  int index = tile_offset + threadIdx.x;
  if (index >= spatial_size) return;

  // Decode the flattened spatial index into (out_h, out_w)
  int out_h = index / out_width;
  int out_w = index % out_width;

  // Initialize the output value with the bias
  float value = bias[o];

  // Loop over input channels
  for (int c = 0; c < in_channels; c++) {
    // Loop over kernel height
    for (int p = 0; p < kernel_size; p++) {
      int h_unscaled = out_h + padding - p * dilation;
      if (h_unscaled % stride != 0) continue;
      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height) continue;
      
      // Loop over kernel width
      for (int q = 0; q < kernel_size; q++) {
        int w_unscaled = out_w + padding - q * dilation;
        if (w_unscaled % stride != 0) continue;
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width) continue;
        
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        value += input[input_idx] * weight[weight_idx];
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
  output[output_idx] = value;
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_spatial_tile(
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
  int kernel_size = weight.size(2); // assume square kernel

  // Calculate output spatial dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int spatial_size = out_height * out_width;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Configure a 3D grid:
  // grid.x: batch index, grid.y: output channel, grid.z: spatial tiling
  int blockSize = 256; // Number of threads per block covering spatial dimension
  dim3 threads(blockSize);
  dim3 grid(batch_size, out_channels, (spatial_size + blockSize - 1) / blockSize);

  conv_transpose2d_forward_kernel_spatial_tile<<<grid, threads>>>(
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
      dilation,
      spatial_size);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
      printf("Error in conv_transpose2d_forward_kernel_spatial_tile: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper to handle the possibility of a None bias
torch::Tensor conv_transpose2d_forward_wrapper_spatial_tile(
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
  
  return conv_transpose2d_forward_cuda_spatial_tile(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_spatial_tile,
        "ConvTranspose2d forward with spatial tiled indexing (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
