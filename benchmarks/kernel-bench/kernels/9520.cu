#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// CUDA kernel for 2D transposed convolution
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

  // Decode index
  int w_out = index % out_width;
  int temp = index / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  float out_val = bias[o];

  #pragma unroll 4
  for (int c = 0; c < in_channels; ++c) {
    #pragma unroll 2
    for (int p = 0; p < kernel_size; ++p) {
      int h_unscaled = h_out + padding - p * dilation;
      bool valid_h = ((h_unscaled % stride) == 0);
      int h_in = valid_h ? (h_unscaled / stride) : 0;
      valid_h = valid_h && (h_in >= 0 && h_in < in_height);

      #pragma unroll 2
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = w_out + padding - q * dilation;
        bool valid_w = ((w_unscaled % stride) == 0);
        int w_in = valid_w ? (w_unscaled / stride) : 0;
        valid_w = valid_w && (w_in >= 0 && w_in < in_width);

        if (valid_h && valid_w) {
          int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
          int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
          out_val += input[input_idx] * weight[weight_idx];
        }
      }
    }
  }

  output[index] = out_val;
}

torch::Tensor conv_transpose2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {

  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  const int in_height = input.size(2);
  const int in_width = input.size(3);
  const int out_channels = weight.size(1);
  const int kernel_size = weight.size(2);
  
  const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  
  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Create CUDA streams
  cudaStream_t streams[2];
  for (int i = 0; i < 2; i++) {
    cudaStreamCreate(&streams[i]);
  }

  const int total_elements = batch_size * out_channels * out_height * out_width;
  const int chunk_size = 1024 * 256; // Process data in chunks
  const int num_chunks = (total_elements + chunk_size - 1) / chunk_size;

  const dim3 block_size(256);
  const dim3 grid_size((chunk_size + block_size.x - 1) / block_size.x);

  int current_buffer = 0;
  for (int chunk = 0; chunk < num_chunks; chunk++) {
    const int chunk_start = chunk * chunk_size;
    const int current_chunk_size = min(chunk_size, total_elements - chunk_start);
    
    // Launch kernel for current chunk
    conv_transpose2d_forward_kernel<<<grid_size, block_size, 0, streams[current_buffer]>>>(
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

    current_buffer = 1 - current_buffer; // Switch buffers
  }

  // Cleanup
  for (int i = 0; i < 2; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  return output;
}

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
        "ConvTranspose2d forward (CUDA) with double buffering",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}