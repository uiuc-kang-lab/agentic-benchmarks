#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Define a maximum kernel size assumed (adjust if necessary)
#define MAX_KERNEL_SIZE 16

// Optimized CUDA kernel for 2D transposed convolution that minimizes warp divergence
// by precomputing valid kernel positions for the h and w dimensions. This avoids divergent
// branches within the inner loops, ensuring a more uniform control flow across threads.

__global__ void conv_transpose2d_forward_kernel_no_divergence(
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

  // Precompute base indices for output location
  int base_h = h_out + padding;
  int base_w = w_out + padding;

  // Precompute valid kernel indices for the h dimension
  int valid_p_count = 0;
  int valid_p[MAX_KERNEL_SIZE];        // stores the valid p index
  int h_in_list[MAX_KERNEL_SIZE];        // stores corresponding h_in
  for (int p = 0; p < kernel_size; p++) {
    int p_dilated = p * dilation;
    // Ensure that the subtraction is non-negative and divisible by stride
    if (base_h >= p_dilated && ((base_h - p_dilated) % stride) == 0) {
      int h_in = (base_h - p_dilated) / stride;
      if (h_in < in_height) {
        valid_p[valid_p_count] = p;
        h_in_list[valid_p_count] = h_in;
        valid_p_count++;
      }
    }
  }

  // Precompute valid kernel indices for the w dimension
  int valid_q_count = 0;
  int valid_q[MAX_KERNEL_SIZE];        // stores the valid q index
  int w_in_list[MAX_KERNEL_SIZE];        // stores corresponding w_in
  for (int q = 0; q < kernel_size; q++) {
    int q_dilated = q * dilation;
    if (base_w >= q_dilated && ((base_w - q_dilated) % stride) == 0) {
      int w_in = (base_w - q_dilated) / stride;
      if (w_in < in_width) {
        valid_q[valid_q_count] = q;
        w_in_list[valid_q_count] = w_in;
        valid_q_count++;
      }
    }
  }

  // Initialize the output value with the bias for channel o using read-only cache
  float out_val = __ldg(&bias[o]);

  // Iterate over input channels
  for (int c = 0; c < in_channels; ++c) {
    // Loop over precomputed valid p positions
    for (int i = 0; i < valid_p_count; i++) {
      int p = valid_p[i];
      int h_in = h_in_list[i];
      // Loop over precomputed valid q positions
      for (int j = 0; j < valid_q_count; j++) {
        int q = valid_q[j];
        int w_in = w_in_list[j];
        
        // Compute flat indices for input and weight tensors
        int input_idx = (((b * in_channels + c) * in_height) + h_in) * in_width + w_in;
        int weight_idx = (((c * out_channels + o) * kernel_size + p) * kernel_size) + q;
        
        // Accumulate contributions using read-only loads
        out_val += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
      }
    }
  }

  // Write the computed result to the output
  int output_idx = (((b * out_channels) + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}

// CUDA forward function using the no-divergence kernel
torch::Tensor conv_transpose2d_forward_cuda_no_divergence(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
  // Get dimensions from input and weight tensors
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);

  // Weight tensor has shape: [in_channels, out_channels, kernel_size, kernel_size]
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);  // assume square kernel

  // Compute the output dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  int total_threads = batch_size * out_channels * out_height * out_width;
  int threads = 128;
  int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_forward_kernel_no_divergence<<<blocks, threads>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_no_divergence: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper function to support bias being None (creates a zero bias tensor if needed)
torch::Tensor conv_transpose2d_forward_wrapper_no_divergence(
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
  return conv_transpose2d_forward_cuda_no_divergence(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_no_divergence,
        "ConvTranspose2d forward (CUDA) with minimized warp divergence",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
