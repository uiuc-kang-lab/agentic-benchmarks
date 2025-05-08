#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <algorithm>

// Inline device function for decoding a flattened index
__device__ inline void decode_index(int index, int out_width, int out_height, int out_channels,
                                      int &b, int &o, int &h_out, int &w_out) {
  w_out = index % out_width;
  int temp = index / out_width;
  h_out = temp % out_height;
  temp /= out_height;
  o = temp % out_channels;
  b = temp / out_channels;
}

// Combined CUDA kernel for conv_transpose2d with chunking, double buffering, and loop unrolling
__global__ void conv_transpose2d_forward_kernel(
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
    int chunk_start,
    int chunk_size) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunk_size) return;

  int index = chunk_start + tid;
  int total = batch_size * out_channels * out_height * out_width;
  if (index >= total) return;

  int b, o, h_out, w_out;
  decode_index(index, out_width, out_height, out_channels, b, o, h_out, w_out);

  // Initialize with bias
  float out_val = bias[o];

  // Process convolution contributions with loop unrolling hints
  #pragma unroll
  for (int c = 0; c < in_channels; ++c) {
    #pragma unroll
    for (int p = 0; p < kernel_size; ++p) {
      int h_unscaled = h_out + padding - p * dilation;
      if (h_unscaled % stride != 0) continue;
      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height) continue;
      #pragma unroll
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = w_out + padding - q * dilation;
        if (w_unscaled % stride != 0) continue;
        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width) continue;

        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        out_val += input[input_idx] * weight[weight_idx];
      }
    }
  }

  output[index] = out_val;
}

// Host launcher using double buffering and chunking
torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
  // Get dimensions
  int batch_size = input.size(0);
  int in_channels = input.size(1);
  int in_height = input.size(2);
  int in_width = input.size(3);

  int out_channels = weight.size(1);
  int kernel_size = weight.size(2); // assume square kernel

  // Calculate output dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  int total_elements = batch_size * out_channels * out_height * out_width;
  int chunk_size = 1024 * 256;  // Process output in sizable chunks
  int num_chunks = (total_elements + chunk_size - 1) / chunk_size;

  // Create two CUDA streams for double buffering
  cudaStream_t streams[2];
  for (int i = 0; i < 2; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  dim3 block(256);
  int current_buffer = 0;

  // Launch kernel per chunk on alternating streams
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int chunk_start = chunk * chunk_size;
    int current_chunk_size = std::min(chunk_size, total_elements - chunk_start);
    dim3 grid((current_chunk_size + block.x - 1) / block.x);

    conv_transpose2d_forward_kernel<<<grid, block, 0, streams[current_buffer]>>>(
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
        chunk_start,
        current_chunk_size);

    current_buffer = 1 - current_buffer;  // Toggle between the two streams
  }

  // Synchronize and destroy streams
  for (int i = 0; i < 2; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in conv_transpose2d_forward_kernel: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Pybind11 wrapper to handle optional bias
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
        "ConvTranspose2d forward (CUDA) with chunking and double buffering",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
