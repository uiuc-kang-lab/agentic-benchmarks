#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define TILE_SIZE 16

//---------------------------------------------------------------------
// Helper Function
//---------------------------------------------------------------------

// Inline device function to check boundary conditions
__device__ inline bool is_within_bounds(int coord, int lower_bound, int upper_bound) {
  return (coord >= lower_bound && coord < upper_bound);
}

//---------------------------------------------------------------------
// Fused Depthwise-Pointwise Convolution Kernel
//---------------------------------------------------------------------

// This kernel fuses depthwise and pointwise convolutions to avoid writing
// an intermediate tensor to global memory. This reduces memory traffic and
// kernel launch overhead while leveraging in-register accumulation.

template <typename scalar_t>
__global__ void fused_depthwise_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,          // [batch, in_channels, in_h, in_w]
    const scalar_t* __restrict__ depthwise_weight, // [in_channels, 1, k, k]
    const scalar_t* __restrict__ depthwise_bias,   // [in_channels] or nullptr
    const scalar_t* __restrict__ pointwise_weight, // [out_channels, in_channels]
    const scalar_t* __restrict__ pointwise_bias,   // [out_channels] or nullptr
    scalar_t* __restrict__ output,                // [batch, out_channels, out_h, out_w]
    int batch,
    int in_channels,
    int out_channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k, int stride, int padding, int dilation) {

  // Decode the batch index and output channel from blockIdx.z
  int linear_idx = blockIdx.z;
  int n = linear_idx / out_channels;
  int oc = linear_idx % out_channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    // Initialize the output with pointwise bias if provided
    scalar_t result = (pointwise_bias != nullptr) ? pointwise_bias[oc] : static_cast<scalar_t>(0);

    // For each input channel, perform the depthwise convolution and accumulate
    // Then apply the pointwise weight multiplication
    for (int c = 0; c < in_channels; ++c) {
      // Start with depthwise bias if provided
      scalar_t dw = (depthwise_bias != nullptr) ? depthwise_bias[c] : static_cast<scalar_t>(0);
      // Iterate over the kernel window
      for (int i = 0; i < k; ++i) {
        int ih = oh * stride - padding + i * dilation;
        bool valid_i = is_within_bounds(ih, 0, in_h);
        for (int j = 0; j < k; ++j) {
          int iw = ow * stride - padding + j * dilation;
          bool valid_j = is_within_bounds(iw, 0, in_w);
          if (valid_i && valid_j) {
            int input_idx = n * (in_channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
            int dw_weight_idx = c * (k * k) + i * k + j;
            dw += input[input_idx] * depthwise_weight[dw_weight_idx];
          }
        }
      }
      // Multiply the computed depthwise result with the pointwise weight
      int pw_weight_idx = oc * in_channels + c;
      result += dw * pointwise_weight[pw_weight_idx];
    }
    
    int output_idx = n * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
    output[output_idx] = result;
  }
}

//---------------------------------------------------------------------
// Core CUDA Forward Function
//---------------------------------------------------------------------

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

  // Depthwise weight shape is expected to be [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_channels = pointwise_weight.size(0);

  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  // Configure grid and block dimensions
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
            (out_h + TILE_SIZE - 1) / TILE_SIZE,
            batch * out_channels);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_depthwise_pointwise_conv2d_cuda", ([&] {
    fused_depthwise_pointwise_conv2d_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        depthwise_bias.defined() && depthwise_bias.numel() > 0 ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
        pointwise_weight.data_ptr<scalar_t>(),
        pointwise_bias.defined() && pointwise_bias.numel() > 0 ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        batch, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        k, stride, padding, dilation);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Fused kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

//---------------------------------------------------------------------
// Helper Function: Convert py::object to at::Tensor
//---------------------------------------------------------------------

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

//---------------------------------------------------------------------
// Wrapper Function
//---------------------------------------------------------------------

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

//---------------------------------------------------------------------
// PyBind11 Module Registration
//---------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "Fused depthwise separable convolution forward with improved efficiency");
}
