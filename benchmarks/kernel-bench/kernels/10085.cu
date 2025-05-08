#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

#define TILE_SIZE 16

// Helper device function: computes the index in a 4D tensor with shape [N, C, H, W]
__device__ __forceinline__ int get4dIndex(int n, int c, int h, int w, int C, int H, int W) {
  return n * C * H * W + c * H * W + h * W + w;
}

// Modular device function for depthwise convolution at a single output location
// Processes one kernel window and returns the computed sum
template <typename scalar_t>
__device__ __forceinline__ scalar_t depthwise_convolve(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int n, int c, int oh, int ow,
    int in_h, int in_w, int channels,
    int k, int stride, int padding, int dilation) {
  scalar_t sum = 0;
  for (int i = 0; i < k; i++) {
    int ih = oh * stride - padding + i * dilation;
    for (int j = 0; j < k; j++) {
      int iw = ow * stride - padding + j * dilation;
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        int input_idx = get4dIndex(n, c, ih, iw, channels, in_h, in_w);
        int weight_idx = c * k * k + i * k + j;
        sum += input[input_idx] * weight[weight_idx];
      }
    }
  }
  return sum;
}

// Modular device function for pointwise convolution at a single output location
// Sums over the input channels
template <typename scalar_t>
__device__ __forceinline__ scalar_t pointwise_convolve(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int n, int oc, int oh, int ow,
    int in_channels, int H, int W) {
  scalar_t sum = 0;
  for (int ic = 0; ic < in_channels; ic++) {
    int input_idx = get4dIndex(n, ic, oh, ow, in_channels, H, W);
    int weight_idx = oc * in_channels + ic;
    sum += input[input_idx] * weight[weight_idx];
  }
  return sum;
}

// Depthwise convolution kernel using modular device functions
// Each thread computes one output element for a given batch and channel
template <typename scalar_t>
__global__ void depthwise_conv2d_modular_kernel(
    const scalar_t* __restrict__ input,    // [batch, channels, in_h, in_w]
    const scalar_t* __restrict__ weight,   // [channels, 1, k, k]
    const scalar_t* __restrict__ bias,     // [channels] or nullptr
    scalar_t* __restrict__ output,         // [batch, channels, out_h, out_w]
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  // Decode batch and channel from gridDim.z
  int linear_idx = blockIdx.z;
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    scalar_t result = depthwise_convolve<scalar_t>(
        input, weight, n, c, oh, ow,
        in_h, in_w, channels, k, stride, padding, dilation);
    if (bias != nullptr) {
      result += bias[c];
    }
    int out_idx = get4dIndex(n, c, oh, ow, channels, out_h, out_w);
    output[out_idx] = result;
  }
}

// Pointwise convolution kernel using modular device functions
// Each thread computes one output element for a given batch and output channel
template <typename scalar_t>
__global__ void pointwise_conv2d_modular_kernel(
    const scalar_t* __restrict__ input,   // [batch, in_channels, H, W] from depthwise output
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, H, W]
    int batch,
    int in_channels,
    int out_channels,
    int H, int W) {

  int linear_idx = blockIdx.z;
  int n = linear_idx / out_channels;
  int oc = linear_idx % out_channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < H && ow < W) {
    scalar_t result = pointwise_convolve<scalar_t>(
        input, weight, n, oc, oh, ow, in_channels, H, W);
    if (bias != nullptr) {
      result += bias[oc];
    }
    int out_idx = get4dIndex(n, oc, oh, ow, out_channels, H, W);
    output[out_idx] = result;
  }
}

// Core CUDA forward function that launches the modular kernels
torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  TORCH_CHECK(depthwise_weight.is_cuda(), "Depthwise weight must be a CUDA tensor");
  TORCH_CHECK(pointwise_weight.is_cuda(), "Pointwise weight must be a CUDA tensor");
  if (depthwise_bias.defined() && depthwise_bias.numel() > 0)
    TORCH_CHECK(depthwise_bias.is_cuda(), "Depthwise bias must be a CUDA tensor if provided");
  if (pointwise_bias.defined() && pointwise_bias.numel() > 0)
    TORCH_CHECK(pointwise_bias.is_cuda(), "Pointwise bias must be a CUDA tensor if provided");

  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  // Launch depthwise convolution kernel
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
            (out_h + TILE_SIZE - 1) / TILE_SIZE,
            batch * in_channels);

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_modular_cuda", ([&] {
    depthwise_conv2d_modular_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
        depthwise_output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        in_h, in_w,
        out_h, out_w,
        k,
        stride,
        padding,
        dilation);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Depthwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // Launch pointwise convolution kernel
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 block_pw(TILE_SIZE, TILE_SIZE);
  dim3 grid_pw((out_w + TILE_SIZE - 1) / TILE_SIZE,
               (out_h + TILE_SIZE - 1) / TILE_SIZE,
               batch * out_channels);

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_modular_cuda", ([&] {
    pointwise_conv2d_modular_kernel<scalar_t><<<grid_pw, block_pw>>>(
        depthwise_output.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        out_channels,
        out_h, out_w);
  }));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper: convert a py::object to an at::Tensor. Supports raw tensors or objects with a 'data' attribute.
at::Tensor toTensor(const py::object& obj) {
  if (obj.is_none()) {
    return at::Tensor();
  }
  try {
    return obj.cast<at::Tensor>();
  } catch (const py::cast_error& e) {
    if (py::hasattr(obj, "data")) {
      return obj.attr("data").cast<at::Tensor>();
    }
    throw std::runtime_error("Expected a torch Tensor or Parameter.");
  }
}

// Wrapper function to handle inputs that may be wrapped in Parameter objects or be None
at::Tensor forward_wrapper(py::object x_obj,
                           py::object depthwise_weight_obj,
                           py::object pointwise_weight_obj,
                           py::object depthwise_bias_obj,
                           py::object pointwise_bias_obj,
                           int stride,
                           int padding,
                           int dilation) {
  auto x = toTensor(x_obj);
  auto depthwise_weight = toTensor(depthwise_weight_obj);
  auto pointwise_weight = toTensor(pointwise_weight_obj);
  auto depthwise_bias = toTensor(depthwise_bias_obj);
  auto pointwise_bias = toTensor(pointwise_bias_obj);

  return forward_cuda(x, depthwise_weight, pointwise_weight,
                      depthwise_bias, pointwise_bias,
                      stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward with modular device functions");
}
