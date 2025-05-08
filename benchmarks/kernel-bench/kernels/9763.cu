#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 16

// Modular device function to compute convolution for a single output pixel
__device__ __forceinline__ float compute_depthwise_conv(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b,
    int c,
    int oh,
    int ow,
    int in_h,
    int in_w,
    int channels,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {
  float sum = 0.0f;
  #pragma unroll
  for (int kh = 0; kh < kernel_h; ++kh) {
    int ih = oh * stride - padding + kh * dilation;
    int iw = ow * stride - padding + kh * dilation;
    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
      // Compute index for the input tensor
      int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
      int weight_idx = c * kernel_h + kh;
      sum += input[input_idx] * weight[weight_idx];
    }
  }
  return sum;
}

// Tiled kernel that processes a tile of the output using the modular device function
__global__ void depthwise_conv2d_modular_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation) {
  // Determine the tile position
  int tile_row = blockIdx.y * TILE_SIZE;
  int tile_col = blockIdx.x * TILE_SIZE;
  
  // Each block in grid.z corresponds to a unique (batch, channel) slice
  int slice = blockIdx.z; // slice index = b * channels + c
  int b = slice / channels;
  int c = slice % channels;
  
  // Each thread processes one output pixel in the tile
  int tx = threadIdx.x % TILE_SIZE;
  int ty = threadIdx.x / TILE_SIZE;
  
  int oh = tile_row + ty;
  int ow = tile_col + tx;

  if (b < batch && oh < out_h && ow < out_w) {
    float sum = compute_depthwise_conv(input, weight, b, c, oh, ow, in_h, in_w, channels, kernel_h, stride, padding, dilation);
    sum += bias[c];
    int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
    output[output_idx] = sum;
  }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  // Ensure input tensors are contiguous
  x = x.contiguous();
  weight = weight.contiguous();

  int batch = x.size(0);
  int channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);
  int kernel_h = weight.size(2); // weight shape: (channels, 1, kernel_h, 1)

  if (groups != channels) {
    throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
  }

  at::Tensor bias_val;
  if (bias.has_value() && bias.value().defined()) {
    bias_val = bias.value().contiguous();
  } else {
    bias_val = at::zeros({channels}, x.options());
  }

  int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - 1) / stride + 1;

  auto output = at::empty({batch, channels, out_h, out_w}, x.options());

  // Define grid dimensions
  dim3 grid(
      (out_w + TILE_SIZE - 1) / TILE_SIZE,
      (out_h + TILE_SIZE - 1) / TILE_SIZE,
      batch * channels);
  // Each block handles a TILE_SIZE x TILE_SIZE tile
  dim3 block(TILE_SIZE * TILE_SIZE);

  depthwise_conv2d_modular_kernel<<<grid, block>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias_val.data_ptr<float>(),
      output.data_ptr<float>(),
      batch,
      channels,
      in_h,
      in_w,
      out_h,
      out_w,
      kernel_h,
      stride,
      padding,
      dilation);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Depthwise 2D Convolution forward (CUDA)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = c10::nullopt,
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"));
}
