#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Kernel using warp-level primitives (__shfl_down_sync) for reduction
// Each warp computes one output pixel by partitioning the work (over input channels and kernel elements) among its 32 lanes
// and then using in-warp shuffle operations to reduce the partial sums without shared memory.

__global__ void conv_transpose2d_forward_kernel_warp(
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

  // Each block is one output pixel. Grid dimensions:
  //   blockIdx.x: output width index
  //   blockIdx.y: output height index
  //   blockIdx.z: encoded as (b * out_channels + o)
  int out_w = blockIdx.x;  // output column index
  int out_h = blockIdx.y;  // output row index
  int bo_idx = blockIdx.z; // combined batch and output channel index
  int o = bo_idx % out_channels;
  int b = bo_idx / out_channels;

  // Use one warp per output pixel; assume blockDim.x == 32
  int lane = threadIdx.x;  // lane id in the warp [0,31]
  float sum = 0.0f;
  
  // Total number of iterations over (c, p, q):
  int total_iter = in_channels * kernel_size * kernel_size;

  // Each thread in the warp processes a subset of (c, p, q) combinations
  for (int iter = lane; iter < total_iter; iter += 32) {
    int tmp = iter;
    int q = tmp % kernel_size;
    tmp /= kernel_size;
    int p = tmp % kernel_size;
    int c = tmp / kernel_size;

    int h_unscaled = out_h + padding - p * dilation;
    if (h_unscaled % stride != 0) continue;
    int h_in = h_unscaled / stride;
    if (h_in < 0 || h_in >= in_height) continue;

    int w_unscaled = out_w + padding - q * dilation;
    if (w_unscaled % stride != 0) continue;
    int w_in = w_unscaled / stride;
    if (w_in < 0 || w_in >= in_width) continue;

    int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
    int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
    sum += input[input_idx] * weight[weight_idx];
  }

  // Perform warp-level reduction using __shfl_down_sync to sum partial results
  unsigned int mask = 0xffffffff;
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }

  // Lane 0 writes the final sum (adding bias) to the output
  if (lane == 0) {
    int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
    output[output_idx] = bias[o] + sum;
  }
}

// Launcher function for the warp-level reduction kernel
torch::Tensor conv_transpose2d_forward_cuda_warp(
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

  // Calculate output dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  
  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Grid: one block per output pixel; each block has 32 threads (one warp)
  dim3 blocks(out_width, out_height, batch_size * out_channels);
  dim3 threads(32);

  conv_transpose2d_forward_kernel_warp<<<blocks, threads>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_warp: %s\n", cudaGetErrorString(err));
  }
  
  return output;
}

// Wrapper function to handle the optional bias argument
torch::Tensor conv_transpose2d_forward_wrapper_warp(
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

  return conv_transpose2d_forward_cuda_warp(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_warp,
        "ConvTranspose2d forward with warp-level reduction (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
