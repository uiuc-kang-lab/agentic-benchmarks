#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Tile size for spatial dimensions
#define TILE_SIZE 16

// Fused kernel combining depthwise and pointwise convolution in one pass
// This kernel avoids writing intermediate depthwise results to global memory
// Input: x [batch, in_channels, in_h, in_w]
// Depthwise weight: [in_channels, 1, k, k] (treated as [in_channels, k, k])
// Pointwise weight: [out_channels, in_channels]
// Depthwise bias: [in_channels] (or nullptr)
// Pointwise bias: [out_channels] (or nullptr)
// Output: [batch, out_channels, out_h, out_w]

// Fused kernel template
template <typename scalar_t>
__global__ void fused_conv2d_kernel(
    const scalar_t* __restrict__ input,         // [batch, in_channels, in_h, in_w]
    const scalar_t* __restrict__ depth_weight,    // [in_channels, 1, k, k] -> treated as [in_channels, k, k]
    const scalar_t* __restrict__ point_weight,    // [out_channels, in_channels]
    const scalar_t* __restrict__ depth_bias,      // [in_channels] or nullptr
    const scalar_t* __restrict__ point_bias,      // [out_channels] or nullptr
    scalar_t* __restrict__ output,                // [batch, out_channels, out_h, out_w]
    int batch,
    int in_channels,
    int out_channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  // Compute spatial coordinates using 2D tiling
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  // Use gridDim.z to encode both batch index and out_channel index
  int linear_idx = blockIdx.z;
  int n = linear_idx / out_channels;
  int oc = linear_idx % out_channels;

  if (ow < out_w && oh < out_h && n < batch) {
    // Initialize the output with the pointwise bias if present
    scalar_t out_val = (point_bias != nullptr) ? point_bias[oc] : static_cast<scalar_t>(0);
    
    // Loop over input channels to fuse depthwise and pointwise computation
    for (int ic = 0; ic < in_channels; ++ic) {
      // Start with depthwise bias if provided
      scalar_t depth_val = (depth_bias != nullptr) ? depth_bias[ic] : static_cast<scalar_t>(0);
      
      // Compute the depthwise convolution for this channel
      for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
          int ih = oh * stride - padding + i * dilation;
          int iw = ow * stride - padding + j * dilation;
          if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = n * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
            int dw_idx = ic * (k * k) + i * k + j;  // Index in depth_weight (flattened over the k x k window)
            depth_val += input[input_idx] * depth_weight[dw_idx];
          }
        }
      }
      
      // Multiply the depthwise result by the pointwise weight and accumulate for the output channel
      int pw_idx = oc * in_channels + ic; // point_weight shape: [out_channels, in_channels]
      out_val += depth_val * point_weight[pw_idx];
    }
    
    // Write the final value to the output tensor
    int output_idx = n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
    output[output_idx] = out_val;
  }
}

// Core CUDA forward function for the fused depthwise separable convolution
torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depth_weight,
    const torch::Tensor& point_weight,
    const torch::Tensor& depth_bias,
    const torch::Tensor& point_bias,
    int stride,
    int padding,
    int dilation) {

  TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
  TORCH_CHECK(depth_weight.is_cuda(), "Depthwise weight must be a CUDA tensor");
  TORCH_CHECK(point_weight.is_cuda(), "Pointwise weight must be a CUDA tensor");

  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);
  
  // Depthwise weight is expected to have shape [in_channels, 1, k, k]
  int k = depth_weight.size(2);

  // Calculate spatial dimensions of output
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Pointwise weight shape: [out_channels, in_channels]
  int out_channels = point_weight.size(0);
  
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  // Define block and grid dimensions for 2D tiling over spatial dims, and use grid.z for batch and output channel
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
            (out_h + TILE_SIZE - 1) / TILE_SIZE,
            batch * out_channels);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_conv2d_cuda", ([&] {
    fused_conv2d_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depth_weight.data_ptr<scalar_t>(),
        point_weight.data_ptr<scalar_t>(),
        (depth_bias.defined() && depth_bias.numel() > 0) ? depth_bias.data_ptr<scalar_t>() : nullptr,
        (point_bias.defined() && point_bias.numel() > 0) ? point_bias.data_ptr<scalar_t>() : nullptr,
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

// Helper function to convert a py::object to a torch::Tensor
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

// Wrapper function to handle potential Parameter objects
at::Tensor forward_wrapper(py::object x_obj,
                           py::object depth_weight_obj,
                           py::object point_weight_obj,
                           py::object depth_bias_obj,
                           py::object point_bias_obj,
                           int stride,
                           int padding,
                           int dilation) {
  auto x = toTensor(x_obj);
  auto depth_weight = toTensor(depth_weight_obj);
  auto point_weight = toTensor(point_weight_obj);
  auto depth_bias = toTensor(depth_bias_obj);
  auto point_bias = toTensor(point_bias_obj);

  return forward_cuda(x, depth_weight, point_weight,
                      depth_bias, point_bias,
                      stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "Fused CUDA depthwise separable convolution");
}
