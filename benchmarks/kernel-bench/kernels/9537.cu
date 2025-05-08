#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// CUDA kernel for 2D transposed convolution with shared memory optimization
__global__ void conv_transpose2d_forward_kernel_shared_mem(
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

  extern __shared__ float shared_mem[];
  
  float* shared_weight = shared_mem; // Block-shared weights
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * out_channels * out_height * out_width;
  if (tid >= total) return;

  // Decode index
  int w_out = tid % out_width;
  int temp = tid / out_width;
  int h_out = temp % out_height;
  temp /= out_height;
  int o = temp % out_channels;
  int b = temp / out_channels;

  // Each thread block loads a portion of the weights into shared memory
  int weight_offset = threadIdx.x;
  int weight_step = blockDim.x;
  for (int idx = weight_offset; idx < in_channels * out_channels * kernel_size * kernel_size; idx += weight_step) {
    shared_weight[idx] = weight[idx];
  }
  __syncthreads();

  float out_val = bias[o];

  #pragma unroll 4
  for (int c = 0; c < in_channels; ++c) {
    #pragma unroll 2
    for (int p = 0; p < kernel_size; ++p) {
      int h_unscaled = h_out + padding - p * dilation;
      if (h_unscaled < 0 || h_unscaled % stride != 0) continue;

      int h_in = h_unscaled / stride;
      if (h_in < 0 || h_in >= in_height) continue;

      #pragma unroll 2
      for (int q = 0; q < kernel_size; ++q) {
        int w_unscaled = w_out + padding - q * dilation;
        if (w_unscaled % stride != 0) continue;

        int w_in = w_unscaled / stride;
        if (w_in < 0 || w_in >= in_width) continue;

        int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
        int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
        out_val += input[input_idx] * shared_weight[weight_idx];
      }
    }
  }

  output[((b * out_channels + o) * out_height + h_out) * out_width + w_out] = out_val;
}

torch::Tensor conv_transpose2d_forward_cuda_shared_mem(
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

  const int total_elements = batch_size * out_channels * out_height * out_width;
  const dim3 block_size(256);
  const dim3 grid_size((total_elements + block_size.x - 1) / block_size.x);

  const int shared_mem_size = in_channels * out_channels * kernel_size * kernel_size * sizeof(float);

  conv_transpose2d_forward_kernel_shared_mem<<<grid_size, block_size, shared_mem_size>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_shared_mem: %s\n", cudaGetErrorString(err));
  }

  return output;
}

torch::Tensor conv_transpose2d_forward_wrapper_shared_mem(
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

  return conv_transpose2d_forward_cuda_shared_mem(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_shared_mem,
        "ConvTranspose2d forward (CUDA) with shared memory optimization",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
