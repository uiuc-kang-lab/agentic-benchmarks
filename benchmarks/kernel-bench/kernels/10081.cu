#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cmath>

namespace py = pybind11;

#define TILE_SIZE 16

// Depthwise convolution kernel remains similar to previous implementations
// Each thread computes one output element for the depthwise conv.

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, channels, in_h, in_w]
    const scalar_t* __restrict__ weight,  // [channels, 1, k, k]
    const scalar_t* __restrict__ bias,    // [channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, channels, out_h, out_w]
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  // grid.z encodes (batch * channels)
  int linear_idx = blockIdx.z;
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    scalar_t sum = (bias != nullptr) ? bias[c] : static_cast<scalar_t>(0);
    for (int i = 0; i < k; ++i) {
      int ih = oh * stride - padding + i * dilation;
      bool valid_ih = (ih >= 0 && ih < in_h);
      for (int j = 0; j < k; ++j) {
        int iw = ow * stride - padding + j * dilation;
        bool valid_iw = (iw >= 0 && iw < in_w);
        if (valid_ih && valid_iw) {
          int input_idx = n * (channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
          int weight_idx = c * (k * k) + i * k + j;
          sum += input[input_idx] * weight[weight_idx];
        }
      }
    }
    int out_idx = n * (channels * out_h * out_w) + c * (out_h * out_w) + oh * out_w + ow;
    output[out_idx] = sum;
  }
}

// Warp-optimized pointwise convolution kernel using warp-level primitives.
// Each block (of 32 threads) collaborates to compute one output pixel's dot product over in_channels.

template <typename scalar_t>
__global__ void warp_optimized_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, in_channels, h, w] (output from depthwise)
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h,
    int w) {

  // Each block computes one output pixel using one warp (blockDim.x == warpSize assumed = 32)
  int total = batch * out_channels * h * w;
  int idx = blockIdx.x;  // one output pixel per block
  if (idx >= total) return;

  // Decode flat index into (n, oc, oh, ow)
  int tmp = idx;
  int ow = tmp % w; tmp /= w;
  int oh = tmp % h; tmp /= h;
  int oc = tmp % out_channels; tmp /= out_channels;
  int n = tmp;

  // Use warp-level reduction to compute dot product over in_channels
  unsigned int mask = 0xffffffff;  // active mask for full warp
  scalar_t sum = 0;

  // Each thread in the warp computes partial sum over a subset of in_channels by striding with warpSize
  for (int ic = threadIdx.x; ic < in_channels; ic += warpSize) {
    int input_idx = n * (in_channels * h * w) + ic * (h * w) + oh * w + ow;
    int weight_idx = oc * in_channels + ic;  // weight layout: [out_channels, in_channels]
    sum += input[input_idx] * weight[weight_idx];
  }

  // Perform warp-level reduction using __shfl_down_sync
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(mask, sum, offset);
  }

  // Lane 0 writes the final result
  if (threadIdx.x == 0) {
    if (bias != nullptr)
      sum += bias[oc];
    int out_idx = n * (out_channels * h * w) + oc * (h * w) + oh * w + ow;
    output[out_idx] = sum;
  }
}

// Core CUDA forward function that applies depthwise then warp-optimized pointwise convolution.

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

  // For depthwise, weight shape is [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Depthwise output: [batch, in_channels, out_h, out_w]
  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  dim3 block_depth(TILE_SIZE, TILE_SIZE);
  dim3 grid_depth((out_w + TILE_SIZE - 1) / TILE_SIZE,
                  (out_h + TILE_SIZE - 1) / TILE_SIZE,
                  batch * in_channels);

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0)
                                     ? depthwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
    depthwise_conv2d_kernel<scalar_t><<<grid_depth, block_depth>>>(
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

  // Pointwise convolution: weight shape: [out_channels, in_channels]
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  // Use warp-level primitives for the pointwise stage.
  // Each block will consist of one warp (32 threads), and one block computes one output pixel.
  int total_output = batch * out_channels * out_h * out_w;
  dim3 block_pw(32);
  dim3 grid_pw(total_output);

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0)
                                     ? pointwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "warp_optimized_pointwise_conv2d_cuda", ([&] {
    warp_optimized_pointwise_conv2d_kernel<scalar_t><<<grid_pw, block_pw>>>(
        depthwise_output.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_weight.data_ptr<scalar_t>()),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        out_channels,
        out_h,
        out_w);
  }));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper: converts a py::object to an at::Tensor. Supports objects with a 'data' attribute.

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

// Wrapper function to handle potential Tensor or Parameter inputs from Python.

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
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward with warp-level pointwise reduction");
}
