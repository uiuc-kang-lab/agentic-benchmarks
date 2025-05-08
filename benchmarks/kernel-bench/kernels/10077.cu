#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdio>

namespace py = pybind11;

// Tile size for 2D spatial decomposition
#define TILE_SIZE 16

// Fused depthwise separable convolution kernel
// This kernel fuses the depthwise and pointwise convolutions to eliminate the overhead
// of writing and reading an intermediate tensor.  The grid is arranged so that grid.z
// maps to (batch * out_channels) and grid.x and grid.y cover the spatial output dimensions.

template <typename scalar_t>
__global__ void fused_conv2d_kernel(
    const scalar_t* __restrict__ input,            // [batch, in_channels, in_h, in_w]
    const scalar_t* __restrict__ depthwise_weight,   // [in_channels, 1, k, k] treated as [in_channels, k, k]
    const scalar_t* __restrict__ depthwise_bias,     // [in_channels] or nullptr
    const scalar_t* __restrict__ pointwise_weight,   // [out_channels, in_channels]
    const scalar_t* __restrict__ pointwise_bias,     // [out_channels] or nullptr
    scalar_t* __restrict__ output,                   // [batch, out_channels, out_h, out_w]
    int batch,
    int in_channels,
    int out_channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  // Map grid z to (batch, out_channel)
  int linear_idx = blockIdx.z;
  int n = linear_idx / out_channels;
  int oc = linear_idx % out_channels;

  // Compute output spatial indices
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    scalar_t result = 0;
    // For each input channel, perform fused depthwise then pointwise computation
    for (int ic = 0; ic < in_channels; ++ic) {
      scalar_t conv = 0;
      // Depthwise convolution: iterate over kernel spatial dimensions
      for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
          int ih = oh * stride - padding + i * dilation;
          int iw = ow * stride - padding + j * dilation;
          if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = n * in_channels * in_h * in_w +
                            ic * in_h * in_w +
                            ih * in_w + iw;
            int weight_idx = ic * k * k + i * k + j;  // depthwise weight index
            conv += input[input_idx] * depthwise_weight[weight_idx];
          }
        }
      }
      // Add depthwise bias if available
      if (depthwise_bias != nullptr) {
        conv += depthwise_bias[ic];
      }
      // Multiply by the corresponding pointwise weight and accumulate
      int pw_idx = oc * in_channels + ic; // pointwise weight index
      result += conv * pointwise_weight[pw_idx];
    }
    // Add pointwise bias if available
    if (pointwise_bias != nullptr) {
      result += pointwise_bias[oc];
    }
    int output_idx = n * out_channels * out_h * out_w +
                     oc * out_h * out_w +
                     oh * out_w + ow;
    output[output_idx] = result;
  }
}

// Host function launching the fused kernel

torch::Tensor fused_forward_cuda(
    const torch::Tensor& x,                  // [batch, in_channels, in_h, in_w]
    const torch::Tensor& depthwise_weight,     // [in_channels, 1, k, k]
    const torch::Tensor& pointwise_weight,     // [out_channels, in_channels]
    const torch::Tensor& depthwise_bias,       // [in_channels] or empty
    const torch::Tensor& pointwise_bias,       // [out_channels] or empty
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

  // Depthwise weight has shape: [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  // Configure 2D grid for spatial dimensions and grid.z for (batch * out_channels)
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
            (out_h + TILE_SIZE - 1) / TILE_SIZE,
            batch * out_channels);

  // Get bias pointers; pass nullptr if bias tensor is empty
  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0)
                                       ? depthwise_bias.data_ptr()
                                       : nullptr;
  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0)
                                       ? pointwise_bias.data_ptr()
                                       : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_conv2d_cuda", ([&] {
    fused_conv2d_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(), // Treated as contiguous [in_channels, k, k]
        reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
        pointwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        out_channels,
        in_h, in_w,
        out_h, out_w,
        k,
        stride,
        padding,
        dilation);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Fused kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper to convert a py::object to an at::Tensor. Supports torch Tensor and objects with a 'data' attribute.

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

// Wrapper function to support both raw tensors and Parameter objects.

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

  return fused_forward_cuda(x, depthwise_weight, pointwise_weight,
                            depthwise_bias, pointwise_bias,
                            stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "Fused CUDA depthwise separable convolution (depthwise and pointwise) to reduce global memory traffic");
}
