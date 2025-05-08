#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// This kernel uses an input-centric approach where each thread processes a single input element
// and computes its contributions to the corresponding output patch. Since different input elements
// may contribute to the same output element, atomicAdd is used when updating global memory.
// To reduce the number of atomic operations, the output tensor is preinitialized with the bias
// values so that the atomic updates only add contributions from the input*weight products.

__global__ void conv_transpose2d_forward_kernel_input_atomic(
    const float* __restrict__ input,
    const float* __restrict__ weight,
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

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * in_channels * in_height * in_width;
  if (tid >= total)
    return;

  // Decode tid into input indices: (b, c, h_in, w_in)
  int w_in = tid % in_width;
  int tmp = tid / in_width;
  int h_in = tmp % in_height;
  tmp /= in_height;
  int c = tmp % in_channels;
  int b = tmp / in_channels;

  float input_val = input[tid];

  // For each kernel spatial offset
  for (int p = 0; p < kernel_size; ++p) {
    int out_h = h_in * stride - padding + p * dilation;
    if (out_h < 0 || out_h >= out_height)
      continue;
    for (int q = 0; q < kernel_size; ++q) {
      int out_w = w_in * stride - padding + q * dilation;
      if (out_w < 0 || out_w >= out_width)
        continue;
      // For every output channel, accumulate the contribution
      for (int o = 0; o < out_channels; ++o) {
        // Weight is stored as [in_channels, out_channels, kernel_size, kernel_size]
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        float w_val = weight[weight_idx];
        float contribution = input_val * w_val;
        int out_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
        atomicAdd(&output[out_idx], contribution);
      }
    }
  }
}

// Launcher for the atomic-based kernel
torch::Tensor conv_transpose2d_forward_cuda_atomic(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation) {

  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  const int in_height = input.size(2);
  const int in_width  = input.size(3);
  const int out_channels = weight.size(1);
  const int kernel_size = weight.size(2);  // assuming square kernel

  // Compute output dimensions
  const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  const int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  int total_input = batch_size * in_channels * in_height * in_width;
  int threads = 256;
  int blocks = (total_input + threads - 1) / threads;

  conv_transpose2d_forward_kernel_input_atomic<<<blocks, threads>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
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
    printf("Error in conv_transpose2d_forward_kernel_input_atomic: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper function that preinitializes the output tensor with the bias and then calls the atomic kernel
torch::Tensor conv_transpose2d_forward_wrapper_atomic(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int dilation) {

  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  const int in_height = input.size(2);
  const int in_width  = input.size(3);
  const int out_channels = weight.size(1);
  const int kernel_size = weight.size(2);
  const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  const int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  // Create output tensor and preinitialize with bias
  torch::Tensor output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());
  torch::Tensor bias;
  if (bias_obj.is(pybind11::none())) {
    bias = torch::zeros({out_channels}, weight.options());
  } else {
    bias = bias_obj.cast<torch::Tensor>();
  }
  // Broadcast bias to the output shape
  output.copy_(bias.view({1, out_channels, 1, 1}).expand({batch_size, out_channels, out_height, out_width}));

  return conv_transpose2d_forward_cuda_atomic(input, weight, output, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_atomic,
        "ConvTranspose2d forward using input-driven atomic updates (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
