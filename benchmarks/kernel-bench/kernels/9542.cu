#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

#define WARP_SIZE 32

// CUDA kernel for 2D transposed convolution using warp-level primitives for reduction.
// Each warp (32 threads) collaborates to compute one output element, distributing the work
// of summing over the input channels and kernel elements. The intra-warp sum is performed
// via __shfl_down_sync, eliminating the need for shared memory in the reduction step.

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

  // Each warp computes one output element.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  int total_output = batch_size * out_channels * out_height * out_width;
  if (warp_id >= total_output) return;

  // Decode warp_id into (b, o, h_out, w_out)
  int w_out = warp_id % out_width;
  int temp = warp_id / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  float sum = 0.0f;

  // Total iterations over input channels and kernel spatial positions
  int total_iter = in_channels * kernel_size * kernel_size;

  // Each lane in the warp computes a partial sum over a subset of iterations
  for (int idx = lane; idx < total_iter; idx += WARP_SIZE) {
    int c = idx / (kernel_size * kernel_size);
    int rem = idx % (kernel_size * kernel_size);
    int p = rem / kernel_size;
    int q = rem % kernel_size;

    int h_unscaled = h_out + padding - p * dilation;
    if (h_unscaled % stride != 0) continue;
    int h_in = h_unscaled / stride;
    if (h_in < 0 || h_in >= in_height) continue;

    int w_unscaled = w_out + padding - q * dilation;
    if (w_unscaled % stride != 0) continue;
    int w_in = w_unscaled / stride;
    if (w_in < 0 || w_in >= in_width) continue;

    int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
    int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;

    sum += input[input_idx] * weight[weight_idx];
  }

  // Warp-level reduction using shuffle down
  unsigned mask = 0xffffffff;
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }

  // Lane 0 adds bias and writes the result.
  if (lane == 0) {
    int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
    output[output_idx] = bias[o] + sum;
  }
}

// CUDA launcher function for conv_transpose2d forward pass using warp-level reduction
torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

  // Input dimensions
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);

  // Weight dimensions: [in_channels, out_channels, kernel_size, kernel_size]
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);  // assuming square kernel

  // Calculate output dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  int total_output = batch_size * out_channels * out_height * out_width;
  // One warp computes one output element; hence total threads = total_output * WARP_SIZE
  int total_threads = total_output * WARP_SIZE;
  int threads = 1024;
  int blocks = (total_threads + threads - 1) / threads;

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

// Wrapper function to manage the possibility of bias being None
torch::Tensor conv_transpose2d_forward_wrapper(
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

  return conv_transpose2d_forward_cuda(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper,
        "ConvTranspose2d forward (CUDA) with warp-level reduction",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
