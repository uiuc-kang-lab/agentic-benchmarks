#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

#define MAX_KERNEL_SIZE 16
#define NUM_STREAMS 4

__global__ void conv_transpose2d_forward_stream_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_start,
    int batch_chunk,
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
  int total = batch_chunk * out_channels * out_height * out_width;
  if (index >= total) return;

  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int local_b = temp / out_channels;
  int b = batch_start + local_b;

  int base_h = h_out + padding;
  int base_w = w_out + padding;

  int valid_p_count = 0;
  int valid_p[MAX_KERNEL_SIZE], h_in_list[MAX_KERNEL_SIZE];
  for (int p = 0; p < kernel_size; p++) {
    int p_dilated = p * dilation;
    if (base_h >= p_dilated && (base_h - p_dilated) % stride == 0) {
      int h_in = (base_h - p_dilated) / stride;
      if (h_in < in_height) {
        valid_p[valid_p_count] = p;
        h_in_list[valid_p_count++] = h_in;
      }
    }
  }

  int valid_q_count = 0;
  int valid_q[MAX_KERNEL_SIZE], w_in_list[MAX_KERNEL_SIZE];
  for (int q = 0; q < kernel_size; q++) {
    int q_dilated = q * dilation;
    if (base_w >= q_dilated && (base_w - q_dilated) % stride == 0) {
      int w_in = (base_w - q_dilated) / stride;
      if (w_in < in_width) {
        valid_q[valid_q_count] = q;
        w_in_list[valid_q_count++] = w_in;
      }
    }
  }

  float out_val = __ldg(&bias[o]);

  for (int c = 0; c < in_channels; ++c) {
    for (int i = 0; i < valid_p_count; i++) {
      int p = valid_p[i];
      int h_in = h_in_list[i];
      for (int j = 0; j < valid_q_count; j++) {
        int q = valid_q[j];
        int w_in = w_in_list[j];
        
        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        
        out_val += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
  output[output_idx] = out_val;
}

torch::Tensor conv_transpose2d_forward_stream(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

  // Extract dimensions
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2);

  // Calculate output dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  
  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Create streams and calculate batch chunks
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; ++i)
    cudaStreamCreate(&streams[i]);

  int chunk_size = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
  int threads = 256;

  // Launch kernels in parallel streams
  for (int i = 0; i < NUM_STREAMS; ++i) {
    int batch_start = i * chunk_size;
    int batch_end = min((i + 1) * chunk_size, batch_size);
    int batch_chunk = batch_end - batch_start;

    if (batch_chunk == 0) continue;

    int total_threads = batch_chunk * out_channels * out_height * out_width;
    int blocks = (total_threads + threads - 1) / threads;

    conv_transpose2d_forward_stream_kernel<<<blocks, threads, 0, streams[i]>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_start,
        batch_chunk,
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
  }

  // Synchronize streams
  for (int i = 0; i < NUM_STREAMS; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  return output;
}

// Wrapper function
TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_stream);
}