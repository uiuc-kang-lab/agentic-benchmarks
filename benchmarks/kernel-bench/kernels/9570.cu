#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

#define MAX_KERNEL_SIZE 16

__global__ void conv_transpose2d_optimized_kernel(
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

  // Calculate output coordinates
  int w_out = index % out_width;
  int h_out = (index / out_width) % out_height;
  int o = (index / (out_width * out_height)) % out_channels;
  int b = index / (out_width * out_height * out_channels);

  // Store valid kernel positions in registers
  int base_h = h_out + padding;
  int base_w = w_out + padding;

  // Precompute valid p indices
  int valid_p[MAX_KERNEL_SIZE], h_in_arr[MAX_KERNEL_SIZE];
  int valid_p_count = 0;
  for (int p = 0; p < kernel_size; p++) {
    int p_offset = p * dilation;
    if (base_h >= p_offset && (base_h - p_offset) % stride == 0) {
      int h_in = (base_h - p_offset) / stride;
      if (h_in < in_height) {
        valid_p[valid_p_count] = p;
        h_in_arr[valid_p_count] = h_in;
        valid_p_count++;
      }
    }
  }

  // Precompute valid q indices
  int valid_q[MAX_KERNEL_SIZE], w_in_arr[MAX_KERNEL_SIZE];
  int valid_q_count = 0;
  for (int q = 0; q < kernel_size; q++) {
    int q_offset = q * dilation;
    if (base_w >= q_offset && (base_w - q_offset) % stride == 0) {
      int w_in = (base_w - q_offset) / stride;
      if (w_in < in_width) {
        valid_q[valid_q_count] = q;
        w_in_arr[valid_q_count] = w_in;
        valid_q_count++;
      }
    }
  }

  float out_val = __ldg(&bias[o]);
  
  // Optimized channel loop with precomputing
  for (int c = 0; c < in_channels; ++c) {
    const int c_out_o = c * out_channels + o;  // Precompute common term
    const int b_in_c = b * in_channels + c;    // Precompute batch-channel index

    #pragma unroll 4
    for (int i = 0; i < valid_p_count; ++i) {
      const int p = valid_p[i];
      const int h_in = h_in_arr[i];
      
      #pragma unroll 4
      for (int j = 0; j < valid_q_count; ++j) {
        const int q = valid_q[j];
        const int w_in = w_in_arr[j];
        
        // Compute indices once
        const int input_idx = (b_in_c * in_height + h_in) * in_width + w_in;
        const int weight_idx = (c_out_o * kernel_size + p) * kernel_size + q;
        
        out_val += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
      }
    }
  }

  output[index] = out_val;
}

torch::Tensor conv_transpose2d_forward_optimized(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
  
  // Dimensions calculation
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);

  // Determine output dimensions
  int out_height = (in_height - 1)*stride - 2*padding + dilation*(kernel_size-1) + 1;
  int out_width = (in_width - 1)*stride - 2*padding + dilation*(kernel_size-1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Launch configuration optimized for H100
  const int threads = 256;  // Better occupancy for large register usage
  int total_threads = output.numel();
  int blocks = (total_threads + threads - 1) / threads;

  conv_transpose2d_optimized_kernel<<<blocks, threads>>>(
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

  cudaDeviceSynchronize();
  return output;
}

// Pybind11 binding remains mostly identical to previous versions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_optimized,
        "Optimized ConvTranspose2D forward",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}