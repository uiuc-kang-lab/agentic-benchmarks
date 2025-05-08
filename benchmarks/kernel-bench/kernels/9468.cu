#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// CUDA kernel using shared memory to load weight values for reduced global memory latency
__global__ void conv_transpose2d_forward_kernel_shared(
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
  // Using 2D thread blocks: threads along output width and height
  int out_w = blockIdx.x * blockDim.x + threadIdx.x;
  int out_h = blockIdx.y * blockDim.y + threadIdx.y;
  // blockIdx.z encodes both batch index and output channel
  int bo_idx = blockIdx.z;
  int o = bo_idx % out_channels;
  int b = bo_idx / out_channels;

  // Declare shared memory for weights. Each block loads a tile of weights for the fixed output channel 'o'.
  extern __shared__ float shared_weight[]; // size: in_channels * kernel_size * kernel_size

  // Total number of weight elements to load
  int weight_count = in_channels * kernel_size * kernel_size;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = tid; i < weight_count; i += blockDim.x * blockDim.y) {
    // Decode linear index into (c, p, q)
    int tmp = i;
    int q = tmp % kernel_size;
    tmp /= kernel_size;
    int p = tmp % kernel_size;
    int c = tmp / kernel_size;
    // Compute the corresponding index in global memory for weight tensor of shape [in_channels, out_channels, kernel, kernel]
    int weight_idx = ((c * out_channels + o) * kernel_size + p) * kernel_size + q;
    shared_weight[i] = weight[weight_idx];
  }
  __syncthreads();

  // Out-of-bound check for output indices
  if (out_w >= out_width || out_h >= out_height)
    return;

  // Begin computation, starting with the bias
  float out_val = bias[o];

  // Loop over input channels and kernel spatial positions
  for (int c = 0; c < in_channels; ++c) {
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
        int weight_shared_idx = c * (kernel_size * kernel_size) + p * kernel_size + q;
        out_val += input[input_idx] * shared_weight[weight_shared_idx];
      }
    }
  }

  int output_idx = ((b * out_channels + o) * out_height + out_h) * out_width + out_w;
  output[output_idx] = out_val;
}

// CUDA launcher function
torch::Tensor conv_transpose2d_forward_cuda_shared(
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

  // Configure 2D thread block and 3D grid arrangement.
  // block: (32, 8) covers output width and height; grid.z covers batch and output channel indices
  dim3 block(32, 8);
  dim3 grid((out_width + block.x - 1) / block.x,
            (out_height + block.y - 1) / block.y,
            batch_size * out_channels);

  // Compute the shared memory size needed for the weight tile
  int weight_count = in_channels * kernel_size * kernel_size;
  size_t shared_mem_size = weight_count * sizeof(float);

  conv_transpose2d_forward_kernel_shared<<<grid, block, shared_mem_size>>>(
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
    printf("Error in conv_transpose2d_forward_kernel_shared: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Wrapper function to handle the possibility of bias being None
torch::Tensor conv_transpose2d_forward_wrapper_shared(
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

  return conv_transpose2d_forward_cuda_shared(input, weight, bias, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_transpose2d_forward_wrapper_shared,
        "ConvTranspose2d forward with shared memory for weight (CUDA)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("stride"),
        pybind11::arg("padding"),
        pybind11::arg("dilation"));
}
