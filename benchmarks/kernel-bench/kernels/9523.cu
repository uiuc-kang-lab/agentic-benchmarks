#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Optimized CUDA kernel combining modular approach and 3D grid mapping
__global__ void optimized_conv_transpose2d_forward_kernel(
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

  // Map threads in 3D: x for output width, y for output height, z for combined batch and channel index
  int w_out = blockIdx.x * blockDim.x + threadIdx.x;
  int h_out = blockIdx.y * blockDim.y + threadIdx.y;
  int batch_channel_idx = blockIdx.z;  // combined batch and out_channel index

  int b = batch_channel_idx / out_channels;  // batch index
  int o = batch_channel_idx % out_channels;  // output channel index

  // Boundary check
  if (b >= batch_size || o >= out_channels || h_out >= out_height || w_out >= out_width)
    return;

  float out_val = bias[o];

  // Shared memory for storing partial results
  extern __shared__ float shared_mem[];
  float *partial_sums = shared_mem;

  // Initialize shared memory with zero
  partial_sums[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;

  // Synchronize threads within this block
  __syncthreads();

  // Iterate over input channels and kernel spatial dimensions
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

        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        partial_sums[threadIdx.y * blockDim.x + threadIdx.x] = __fmaf_rn(input[input_idx], weight[weight_idx], partial_sums[threadIdx.y * blockDim.x + threadIdx.x]);
      }
    }
  }

  // Synchronize to ensure all threads have computed their partial sums
  __syncthreads();

  // Reduce partial sums within this block
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    for (int i = 0; i < blockDim.x * blockDim.y; ++i) {
      out_val += partial_sums[i];
    }
    int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
    output[output_idx] = out_val;
  }
}

// CUDA launcher function
torch::Tensor optimized_conv_transpose2d_forward_cuda(
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
  int in_width  = input.size(3);

  // Weight dimensions: [in_channels, out_channels, kernel_size, kernel_size]
  int out_channels = weight.size(1);
  int kernel_size = weight.size(2); // assuming square kernel

  // Compute output dimensions
  int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
  int out_width  = (in_width  - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

  auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

  // Use 3D grid: x for width, y for height, z for combined batch and channel
  dim3 block(16, 16);
  dim3 grid((out_width + block.x - 1) / block.x,
            (out_height + block.y - 1) / block.y,
            batch_size * out_channels);

  size_t shared_mem_size = block.x * block.y * sizeof(float);

  optimized_conv_transpose2d_forward_kernel<<<grid, block, shared_mem_size>>>(
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
    printf("Error in optimized_conv_transpose2d_forward_kernel: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper function to handle bias being None
torch::Tensor optimized_conv_transpose2d_forward_wrapper(
    torch::Tensor input,
    torch::Tensor weight,
    pybind11::object bias_obj,  // accepts None
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

  return optimized_conv_transpose2d_forward_cuda(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &optimized_conv_transpose2d_forward_wrapper,
        "Optimized ConvTranspose2d forward (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
