#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Use block dimensions that align with warp size for coalesced accesses
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

// Depthwise convolution kernel with aligned, coalesced global memory accesses
// Uses __ldg intrinsic to load from global memory in a read-only fashion

template <typename scalar_t>
__global__ void coalesced_aligned_depthwise_conv2d_kernel(
    const scalar_t * __restrict__ input,    // [batch, channels, in_h, in_w]
    const scalar_t * __restrict__ weight,   // [channels, 1, k, k]
    const scalar_t * __restrict__ bias,     // [channels] or nullptr
    scalar_t * __restrict__ output,         // [batch, channels, out_h, out_w]
    int batch, 
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k, 
    int stride, 
    int padding, 
    int dilation) {

  // Each block in grid.z corresponds to one (batch, channel)
  int linear_idx = blockIdx.z;
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  // Compute output spatial coordinates
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    scalar_t sum = (bias != nullptr) ? bias[c] : static_cast<scalar_t>(0);
    // Loop over kernel window
    for (int i = 0; i < k; i++) {
      int ih = oh * stride - padding + i * dilation;
      if (ih >= 0 && ih < in_h) {
        for (int j = 0; j < k; j++) {
          int iw = ow * stride - padding + j * dilation;
          if (iw >= 0 && iw < in_w) {
            int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
            int weight_idx = c * k * k + i * k + j;
            // Use __ldg for read-only, aligned memory access for coalescing
            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
          }
        }
      }
    }
    int output_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
  }
}

// Pointwise convolution kernel with aligned, coalesced global memory accesses
// Threads in a warp access consecutive output columns ensuring coalescing

template <typename scalar_t>
__global__ void coalesced_aligned_pointwise_conv2d_kernel(
    const scalar_t * __restrict__ input,   // [batch, in_channels, h, w] (depthwise output)
    const scalar_t * __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t * __restrict__ bias,    // [out_channels] or nullptr
    scalar_t * __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h, int w) {

  // Each block in grid.z corresponds to one (batch, out_channel)
  int linear_idx = blockIdx.z;
  int n = linear_idx / out_channels;
  int oc = linear_idx % out_channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < h && ow < w) {
    scalar_t sum = (bias != nullptr) ? bias[oc] : static_cast<scalar_t>(0);
    for (int ic = 0; ic < in_channels; ++ic) {
      int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
      int weight_idx = oc * in_channels + ic;
      sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
    }
    int output_idx = n * out_channels * h * w + oc * h * w + oh * w + ow;
    output[output_idx] = sum;
  }
}

// Core CUDA forward function

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

  // Depthwise weight shape: [channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid((out_w + block.x - 1) / block.x,
            (out_h + block.y - 1) / block.y,
            batch * in_channels);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "coalesced_aligned_depthwise_conv2d_cuda", ([&] {
    coalesced_aligned_depthwise_conv2d_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
        depthwise_output.data_ptr<scalar_t>(),
        batch, in_channels,
        in_h, in_w,
        out_h, out_w,
        k, stride, padding, dilation);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Depthwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 block_pw(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid_pw((out_w + block_pw.x - 1) / block_pw.x,
                 (out_h + block_pw.y - 1) / block_pw.y,
                 batch * out_channels);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "coalesced_aligned_pointwise_conv2d_cuda", ([&] {
    coalesced_aligned_pointwise_conv2d_kernel<scalar_t><<<grid_pw, block_pw>>>(
        depthwise_output.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
        output.data_ptr<scalar_t>(),
        batch, in_channels, out_channels, out_h, out_w);
  }));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper: convert a py::object to an at::Tensor. Supports direct tensors or objects with a 'data' attribute.

at::Tensor toTensor(const py::object& obj) {
  if (obj.is_none()) {
    return at::Tensor();
  }
  try {
    return obj.cast<at::Tensor>();
  } catch (const py::cast_error &e) {
    if (py::hasattr(obj, "data")) {
      return obj.attr("data").cast<at::Tensor>();
    }
    throw std::runtime_error("Expected a torch Tensor or Parameter.");
  }
}

// Wrapper function to handle inputs. Expected signature: forward(tensor, tensor, tensor, tensor, tensor, int, int, int) -> tensor

at::Tensor forward_wrapper(py::object x_obj,
                           py::object depthwise_weight_obj,
                           py::object pointwise_weight_obj,
                           py::object depthwise_bias_obj,
                           py::object pointwise_bias_obj,
                           int stride,
                           int padding,
                           int dilation) {
  auto x = toTensor(x_obj);
  auto dw = toTensor(depthwise_weight_obj);
  auto pw = toTensor(pointwise_weight_obj);
  auto db = toTensor(depthwise_bias_obj);
  auto pb = toTensor(pointwise_bias_obj);

  return forward_cuda(x, dw, pw, db, pb, stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "CUDA aligned coalesced depthwise separable convolution forward");
}
