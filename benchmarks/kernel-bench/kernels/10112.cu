#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4

// Use a stride loop to handle large workloads
__global__ void depthwise_conv2d_kernel_stride(const float* __restrict__ input,
                                                const float* __restrict__ weight,
                                                const float* __restrict__ bias,
                                                float* __restrict__ output,
                                                int batch,
                                                int channels,
                                                int in_h, int in_w,
                                                int out_h, int out_w,
                                                int k,
                                                int stride,
                                                int padding,
                                                int dilation) {
  int total_outputs = batch * channels * out_h * out_w;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_index = blockIdx.y * gridDim.x * blockDim.x + index;

  while (stride_index < total_outputs) {
    int ow = stride_index % out_w;
    int tmp = stride_index / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int c = tmp % channels;
    int n = tmp / channels;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < k; ++j) {
        int ih = oh * stride - padding + i * dilation;
        int iw = ow * stride - padding + j * dilation;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
          int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
          int weight_idx = c * k * k + i * k + j;
          sum += input[input_idx] * weight[weight_idx];
        }
      }
    }
    if (bias != nullptr)
      sum += bias[c];
    output[stride_index] = sum;

    stride_index += gridDim.y * gridDim.x * blockDim.x;
  }
}

__global__ void pointwise_conv2d_kernel_stride(const float* __restrict__ input,
                                                const float* __restrict__ weight,
                                                const float* __restrict__ bias,
                                                float* __restrict__ output,
                                                int batch,
                                                int in_channels,
                                                int out_channels,
                                                int h,
                                                int w) {
  int total_outputs = batch * out_channels * h * w;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_index = blockIdx.y * gridDim.x * blockDim.x + index;

  while (stride_index < total_outputs) {
    int ow = stride_index % w;
    int tmp = stride_index / w;
    int oh = tmp % h;
    tmp = tmp / h;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
      int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
      int weight_idx = oc * in_channels + ic;
      sum += input[input_idx] * weight[weight_idx];
    }
    if (bias != nullptr)
      sum += bias[oc];
    output[stride_index] = sum;

    stride_index += gridDim.y * gridDim.x * blockDim.x;
  }
}

// Core CUDA aggregation function
torch::Tensor forward_cuda(const torch::Tensor& x,
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
  int blocks_x = (total_depthwise + threads - 1) / threads;
  dim3 blocks(blocks_x, 1);  // Ensure larger workloads split into more y-dims

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0)
                                     ? depthwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
    depthwise_conv2d_kernel_stride<<<blocks, threads>>>(
        x.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        reinterpret_cast<const float*>(depthwise_bias_ptr),
        depthwise_output.data_ptr<float>(),
        batch,
        in_channels,
        in_h, in_w,
        out_h, out_w,
        k,
        stride,
        padding,
        dilation);
  }));

  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  int total_pointwise = batch * out_channels * out_h * out_w;
  int blocks_x_p = (total_pointwise + threads - 1) / threads;
  dim3 blocks_p(blocks_x_p, 1);

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0)
                                     ? pointwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
    pointwise_conv2d_kernel_stride<<<blocks_p, threads>>>(
        depthwise_output.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        reinterpret_cast<const float*>(pointwise_bias_ptr),
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        out_h, out_w);
  }));

  return output;
}

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
