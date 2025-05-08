#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Define maximum constant memory sizes (in number of elements)
// For float: 16384 elements * 4 bytes = 65536 bytes (64KB)
// For double: 16384 elements * 8 bytes = 131072 bytes (128KB) - adjust as needed
#define MAX_DEPTHWISE_WEIGHT_SIZE 16384
#define MAX_POINTWISE_WEIGHT_SIZE 16384

// Declare constant memory for float and double weights
__constant__ float c_depthwise_weight_float[MAX_DEPTHWISE_WEIGHT_SIZE];
__constant__ float c_pointwise_weight_float[MAX_POINTWISE_WEIGHT_SIZE];

__constant__ double c_depthwise_weight_double[MAX_DEPTHWISE_WEIGHT_SIZE];
__constant__ double c_pointwise_weight_double[MAX_POINTWISE_WEIGHT_SIZE];

// Device helper functions for accessing constant memory based on type

// For depthwise weight
template <typename scalar_t>
__device__ inline scalar_t get_depthwise_weight(int idx);

template <>
__device__ inline float get_depthwise_weight<float>(int idx) {
    return c_depthwise_weight_float[idx];
}

template <>
__device__ inline double get_depthwise_weight<double>(int idx) {
    return c_depthwise_weight_double[idx];
}

// For pointwise weight
template <typename scalar_t>
__device__ inline scalar_t get_pointwise_weight(int idx);

template <>
__device__ inline float get_pointwise_weight<float>(int idx) {
    return c_pointwise_weight_float[idx];
}

template <>
__device__ inline double get_pointwise_weight<double>(int idx) {
    return c_pointwise_weight_double[idx];
}

// Depthwise convolution kernel using constant memory for weights
// input: [batch, channels, in_h, in_w]
// bias: [channels] or nullptr
// output: [batch, channels, out_h, out_w]
// depthwise weights are stored in constant memory

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel_const(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * channels * out_h * out_w;
  if (index >= total) return;

  // Decode index into (n, c, oh, ow)
  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int c = tmp % channels;
  int n = tmp / channels;

  scalar_t sum = 0;
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      int ih = oh * stride - padding + i * dilation;
      int iw = ow * stride - padding + j * dilation;
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        int input_idx = n * (channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw;
        int weight_idx = c * k * k + i * k + j;
        sum += input[input_idx] * get_depthwise_weight<scalar_t>(weight_idx);
      }
    }
  }
  if (bias != nullptr) sum += bias[c];
  output[index] = sum;
}

// Pointwise convolution kernel using constant memory for weights
// input: [batch, in_channels, h, w] (output of depthwise conv)
// bias: [out_channels] or nullptr
// output: [batch, out_channels, h, w]
// pointwise weights are stored in constant memory and are of shape [out_channels, in_channels]

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel_const(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int h, int w) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * h * w;
  if (index >= total) return;

  // Decode index into (n, oc, oh, ow)
  int ow = index % w;
  int tmp = index / w;
  int oh = tmp % h;
  tmp = tmp / h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  scalar_t sum = 0;
  for (int ic = 0; ic < in_channels; ++ic) {
    int input_idx = n * (in_channels * h * w) + ic * (h * w) + oh * w + ow;
    int weight_idx = oc * in_channels + ic;
    sum += input[input_idx] * get_pointwise_weight<scalar_t>(weight_idx);
  }
  if (bias != nullptr) sum += bias[oc];
  output[index] = sum;
}

// Forward CUDA function that loads weights into constant memory and launches kernels

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

  // Depthwise weight expected shape: [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Pointwise weight shape: [out_channels, in_channels]
  int out_channels = pointwise_weight.size(0);

  // Check that the weight sizes do not exceed constant memory limits
  int depthwise_count = depthwise_weight.numel(); // in_channels * k * k
  int pointwise_count = pointwise_weight.numel();   // out_channels * in_channels
  TORCH_CHECK(depthwise_count <= MAX_DEPTHWISE_WEIGHT_SIZE, "Depthwise weights exceed constant memory capacity");
  TORCH_CHECK(pointwise_count <= MAX_POINTWISE_WEIGHT_SIZE, "Pointwise weights exceed constant memory capacity");

  // Copy weights to constant memory based on scalar type
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "copy_to_constant", ([&] {
    if (std::is_same<scalar_t, float>::value) {
      cudaMemcpyToSymbol(c_depthwise_weight_float, depthwise_weight.data_ptr<scalar_t>(), depthwise_count * sizeof(scalar_t));
      cudaMemcpyToSymbol(c_pointwise_weight_float, pointwise_weight.data_ptr<scalar_t>(), pointwise_count * sizeof(scalar_t));
    } else {
      cudaMemcpyToSymbol(c_depthwise_weight_double, depthwise_weight.data_ptr<scalar_t>(), depthwise_count * sizeof(scalar_t));
      cudaMemcpyToSymbol(c_pointwise_weight_double, pointwise_weight.data_ptr<scalar_t>(), pointwise_count * sizeof(scalar_t));
    }
  }));

  // Allocate tensor for depthwise convolution output
  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  // Launch depthwise convolution kernel
  int total_depthwise = batch * in_channels * out_h * out_w;
  int threads = 256;
  int blocks = (total_depthwise + threads - 1) / threads;

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_const_cuda", ([&] {
    depthwise_conv2d_kernel_const<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
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
  
  // Allocate tensor for pointwise convolution output
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
  int total_pointwise = batch * out_channels * out_h * out_w;
  blocks = (total_pointwise + threads - 1) / threads;
  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_const_cuda", ([&] {
    pointwise_conv2d_kernel_const<scalar_t><<<blocks, threads>>>(
        depthwise_output.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        out_channels,
        out_h, out_w);
  }));

  return output;
}

// Helper function to convert a py::object to an at::Tensor
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

// Wrapper function callable from Python
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
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward using constant memory");
}
