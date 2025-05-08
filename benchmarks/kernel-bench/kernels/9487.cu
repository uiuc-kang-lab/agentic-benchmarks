#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Device helper: compute the contribution from one input channel using cached (shared) weight
__device__ inline float compute_shared_contribution(
    const float* __restrict__ input,
    const float* __restrict__ shared_weight,
    int b,
    int c,
    int out_h,
    int out_w,
    int in_height,
    int in_width,
    int in_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
  float sum = 0.0f;
  for (int p = 0; p < kernel_size; ++p) {
    int h_unscaled = out_h + padding - p * dilation;
    if (h_unscaled % stride != 0)
      continue;
    int h_in = h_unscaled / stride;
    if (h_in < 0 || h_in >= in_height)
      continue;
    for (int q = 0; q < kernel_size; ++q) {
      int w_unscaled = out_w + padding - q * dilation;
      if (w_unscaled % stride != 0)
        continue;
      int w_in = w_unscaled / stride;
      if (w_in < 0 || w_in >= in_width)
        continue;
      int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
      int weight_idx = c * (kernel_size * kernel_size) + p * kernel_size + q;
      sum += input[input_idx] * shared_weight[weight_idx];
    }
  }
  return sum;
}

// Combined CUDA kernel: uses 2D thread blocks to cover spatial output and caches weight for the current output channel in shared memory
__global__ void conv_transpose2d_forward_kernel_combined(
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

  // Determine output spatial coordinates
  int out_w = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = blockIdx.y * blockDim.y + threadIdx.y;

  // blockIdx.z encodes both batch index and output channel
  int bo_idx = blockIdx.z;
  int o = bo_idx % out_channels;
  int b = bo_idx / out_channels;

  // Boundary check
  if (out_w >= out_width || out_h >= out_height)
    return;

  // Declare shared memory for weight tile for the current output channel.
  // The tile size is: in_channels * kernel_size * kernel_size floats.
  extern __shared__ float shared_weight[];
  int weight_elements = in_channels * kernel_size * kernel_size;

  // Each thread in the block cooperatively loads part of the weight
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < weight_elements; i += blockDim.x * blockDim.y) {
    int tmp = i;
    int q = tmp % kernel_size;
    tmp /= kernel_size;
    int p = tmp % kernel_size;
    int c = tmp / kernel_size;
    // Global weight index: weight layout is [in_channels, out_channels, kernel, kernel]
    int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
    shared_weight[i] = weight[weight_idx];
  }
  __syncthreads();

  // Start with bias for the current output channel
  float out_val = bias[o];

  // Loop over input channels and accumulate contributions using the cached weights
  for (int c = 0; c < in_channels; ++c) {
    out_val += compute_shared_contribution(input, shared_weight, b, c, out_h, out_w,
                                             in_height, in_width, in_channels,
                                             kernel_size, stride, padding, dilation);
  }

  // Write result back to the output tensor
  int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
  output[output_idx] = out_val;
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_combined(
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

  // Configure 2D thread blocks for spatial dimensions.
  // Each block will handle a tile of the output spatial domain, while grid z handles (batch, out_channel).
  dim3 block(32, 8);
  dim3 grid((out_width + block.x - 1) / block.x,
            (out_height + block.y - 1) / block.y,
            batch_size * out_channels);

  // Shared memory size: weight tile for one output channel
  int weight_elements = in_channels * kernel_size * kernel_size;
  size_t shared_mem_size = weight_elements * sizeof(float);

  conv_transpose2d_forward_kernel_combined<<<grid, block, shared_mem_size>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_combined: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper function to handle optional bias
torch::Tensor conv_transpose2d_forward_wrapper_combined(
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
  
  return conv_transpose2d_forward_cuda_combined(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_combined,
        "ConvTranspose2d forward combined kernel (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
