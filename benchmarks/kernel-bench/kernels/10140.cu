#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256

// Depthwise convolution kernel with a stride loop for large workloads
template <typename scalar_t>
__global__ void depthwise_conv2d_kernel_stride(
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

  int total = batch * channels * out_h * out_w;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;  // stride loop step

  while (idx < total) {
    int temp = idx;
    int ow = temp % out_w; temp /= out_w;
    int oh = temp % out_h; temp /= out_h;
    int c  = temp % channels;
    int n  = temp / channels;

    scalar_t sum = 0;
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    // Iterate over kernel window
    #pragma unroll
    for (int i = 0; i < k; i++) {
      #pragma unroll
      for (int j = 0; j < k; j++) {
        int ih = h_start + i * dilation;
        int iw = w_start + j * dilation;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
          int channel_input_stride = in_h * in_w;
    int input_batch_stride = channels * channel_input_stride;
    int input_index = n * input_batch_stride + c * channel_input_stride + ih * in_w + iw;
          int weight_index = c * k * k + i * k + j;
          sum += input[input_index] * weight[weight_index];
        }
      }
    }

    if (bias != nullptr) {
      sum += bias[c];
    }
    output[idx] = sum;

    idx += step;
  }
}

// Pointwise (1x1) convolution kernel with a stride loop for large workloads
template <typename scalar_t>
__global__ void pointwise_conv2d_kernel_stride(
    const scalar_t* __restrict__ input,    // [batch, in_channels, h, w]
    const scalar_t* __restrict__ weight,   // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,     // [out_channels] or nullptr
    scalar_t* __restrict__ output,         // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h, int w) {

  int total = batch * out_channels * h * w;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;  // stride loop step

  while (idx < total) {
    int temp = idx;
    int ow = temp % w; temp /= w;
    int oh = temp % h; temp /= h;
    int oc = temp % out_channels;
    int n  = temp / out_channels;

    scalar_t sum = 0;
    for (int ic = 0; ic < in_channels; ic++) {
      int input_index = n * in_channels * h * w + ic * h * w + oh * w + ow;
      int weight_index = oc * in_channels + ic;
      sum += input[input_index] * weight[weight_index];
    }
    if (bias != nullptr) {
      sum += bias[oc];
    }
    output[idx] = sum;

    idx += step;
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
  
  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);
  
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  int total_depthwise = batch * in_channels * out_h * out_w;
  int threads = THREADS_PER_BLOCK;
  int blocks = (total_depthwise + threads - 1) / threads;

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0)
                                     ? depthwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
    depthwise_conv2d_kernel_stride<scalar_t><<<blocks, threads>>>(
      x.data_ptr<scalar_t>(),
      depthwise_weight.data_ptr<scalar_t>(),
      reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
      depthwise_output.data_ptr<scalar_t>(),
      batch, in_channels, in_h, in_w,
      out_h, out_w, k, stride, padding, dilation);
  }));

  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
  
  int total_pointwise = batch * out_channels * out_h * out_w;
  int blocks_pointwise = (total_pointwise + threads - 1) / threads;

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0)
                                     ? pointwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
    pointwise_conv2d_kernel_stride<scalar_t><<<blocks_pointwise, threads>>>(
      depthwise_output.data_ptr<scalar_t>(),
      pointwise_weight.data_ptr<scalar_t>(),
      reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
      output.data_ptr<scalar_t>(),
      batch, in_channels, out_channels, out_h, out_w);
  }));

  return output;
}

// Helper: Convert py::object to at::Tensor
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

// Wrapper function
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
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}
