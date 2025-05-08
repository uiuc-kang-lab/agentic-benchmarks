#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace py = pybind11;

// Block dimensions
#define TILE_SIZE 16

// Maximum elements for weight arrays in constant memory (in floats/doubles)
// Assumes that the weight tensors are small enough to fit in constant memory (max 64KB total)
#define MAX_DW_WEIGHT 16384
#define MAX_PW_WEIGHT 16384

// Declare constant memory for depthwise and pointwise weights (float and double versions)
__constant__ float const_depthwise_weight_float[MAX_DW_WEIGHT];
__constant__ float const_pointwise_weight_float[MAX_PW_WEIGHT];

__constant__ double const_depthwise_weight_double[MAX_DW_WEIGHT];
__constant__ double const_pointwise_weight_double[MAX_PW_WEIGHT];

// Helper functions to access constant memory weights
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_const_depthwise_weight(int idx);

template <>
__device__ __forceinline__ float get_const_depthwise_weight<float>(int idx) {
    return const_depthwise_weight_float[idx];
}

template <>
__device__ __forceinline__ double get_const_depthwise_weight<double>(int idx) {
    return const_depthwise_weight_double[idx];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_const_pointwise_weight(int idx);

template <>
__device__ __forceinline__ float get_const_pointwise_weight<float>(int idx) {
    return const_pointwise_weight_float[idx];
}

template <>
__device__ __forceinline__ double get_const_pointwise_weight<double>(int idx) {
    return const_pointwise_weight_double[idx];
}

// Depthwise convolution kernel using constant memory for weights
// Input: [batch, channels, in_h, in_w]
// Weight: stored in constant memory, layout: [channels, 1, k, k]
// Bias: [channels] or nullptr
// Output: [batch, channels, out_h, out_w]

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,  // may be nullptr
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

  // Each block z-dimension corresponds to one (batch, channel) pair
  int linear_idx = blockIdx.z;
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  if (oh < out_h && ow < out_w) {
    scalar_t sum = (bias != nullptr) ? bias[c] : static_cast<scalar_t>(0);
    for (int i = 0; i < k; ++i) {
      int ih = oh * stride - padding + i * dilation;
      bool ih_valid = (ih >= 0 && ih < in_h);
      for (int j = 0; j < k; ++j) {
        int iw = ow * stride - padding + j * dilation;
        bool iw_valid = (iw >= 0 && iw < in_w);
        int valid = ih_valid && iw_valid;
        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
        int weight_idx = c * k * k + i * k + j;
        scalar_t w = get_const_depthwise_weight<scalar_t>(weight_idx);
        scalar_t in_val = valid ? input[input_idx] : static_cast<scalar_t>(0);
        sum += in_val * w;
      }
    }
    int output_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
  }
}

// Pointwise convolution kernel using constant memory for weights
// Input: [batch, in_channels, h, w] (output from depthwise convolution)
// Weight: stored in constant memory, layout: [out_channels, in_channels]
// Bias: [out_channels] or nullptr
// Output: [batch, out_channels, h, w]

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,  // may be nullptr
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int h, int w) {

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
      scalar_t w = get_const_pointwise_weight<scalar_t>(weight_idx);
      sum += input[input_idx] * w;
    }
    int output_idx = n * out_channels * h * w + oc * h * w + oh * w + ow;
    output[output_idx] = sum;
  }
}

// Core CUDA forward function with constant memory weight loading
// Performs depthwise separable convolution by launching two kernels

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

  // Depthwise weight shape: [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  // Copy weight tensors to constant memory
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "copy_to_const", ([&] {
    using scalar_t = typename std::remove_reference<decltype(x.scalar_type())>::type;
    if (std::is_same<scalar_t, float>::value) {
      cudaMemcpyToSymbol(const_depthwise_weight_float,
                           depthwise_weight.data_ptr<scalar_t>(),
                           depthwise_weight.numel() * sizeof(scalar_t));
      cudaMemcpyToSymbol(const_pointwise_weight_float,
                           pointwise_weight.data_ptr<scalar_t>(),
                           pointwise_weight.numel() * sizeof(scalar_t));
    } else {
      cudaMemcpyToSymbol(const_depthwise_weight_double,
                           depthwise_weight.data_ptr<scalar_t>(),
                           depthwise_weight.numel() * sizeof(scalar_t));
      cudaMemcpyToSymbol(const_pointwise_weight_double,
                           pointwise_weight.data_ptr<scalar_t>(),
                           pointwise_weight.numel() * sizeof(scalar_t));
    }
  }));

  // Launch depthwise convolution kernel
  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
            (out_h + TILE_SIZE - 1) / TILE_SIZE,
            batch * in_channels);

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv_const_cuda", ([&] {
    depthwise_conv2d_kernel<scalar_t><<<grid, block>>>(
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

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv_const_cuda", ([&] {
    pointwise_conv2d_kernel<scalar_t><<<grid_pw, block_pw>>>(
        depthwise_output.data_ptr<scalar_t>(),
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

// Helper to convert a py::object to an at::Tensor.
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

// Wrapper function exposed to Python
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
  m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward using constant memory for weights");
}
