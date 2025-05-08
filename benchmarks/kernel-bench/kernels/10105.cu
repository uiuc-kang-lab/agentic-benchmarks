#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <cstdio>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256
// Define maximum number of elements in constant memory for depthwise weights
// Adjust this value if needed, but it must be within the hardware constant memory limits.
#define MAX_DEPTHWISE_WEIGHT_SIZE 16384

// Declare constant memory for depthwise weights for float and double types
__constant__ float depthwise_weight_const_f[MAX_DEPTHWISE_WEIGHT_SIZE];
__constant__ double depthwise_weight_const_d[MAX_DEPTHWISE_WEIGHT_SIZE];

// Template helper to get pointer to constant memory weights based on type
template<typename scalar_t>
__device__ __forceinline__ const scalar_t* get_depthwise_weight_const();

template<>
__device__ __forceinline__ const float* get_depthwise_weight_const<float>() {
    return depthwise_weight_const_f;
}

template<>
__device__ __forceinline__ const double* get_depthwise_weight_const<double>() {
    return depthwise_weight_const_d;
}

// Optimized depthwise convolution kernel using constant memory for weights.
// The weight data is read from constant memory via get_depthwise_weight_const<scalar_t>().
template <typename scalar_t>
__global__ void optimized_depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, channels, in_h, in_w]
    // Note: weight is now in constant memory, so it is not passed as a parameter.
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

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * channels * out_h * out_w;
  if (index >= total)
      return;

  // Decode flat index into (n, c, oh, ow)
  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int c = tmp % channels;
  int n = tmp / channels;

  scalar_t sum = 0;
  const scalar_t* weight_const = get_depthwise_weight_const<scalar_t>();

  // Loop over kernel dimensions and accumulate convolution result
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      int ih = oh * stride - padding + i * dilation;
      int iw = ow * stride - padding + j * dilation;
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
        int weight_idx = c * k * k + i * k + j;
        sum += input[input_idx] * weight_const[weight_idx];
      }
    }
  }
  if (bias != nullptr)
    sum += bias[c];
  output[index] = sum;
}

// Pointwise (1x1) convolution kernel remains unchanged.
template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, in_channels, h, w]
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h,
    int w) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * h * w;
  if (index >= total)
      return;

  // Decode flat index into (n, oc, oh, ow)
  int ow = index % w;
  int tmp = index / w;
  int oh = tmp % h;
  tmp = tmp / h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  scalar_t sum = 0;
  for (int ic = 0; ic < in_channels; ++ic) {
    int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
    int weight_idx = oc * in_channels + ic;
    sum += input[input_idx] * weight[weight_idx];
  }
  if (bias != nullptr)
    sum += bias[oc];
  output[index] = sum;
}

// Core CUDA forward function with constant memory optimization for depthwise weights.
// It copies the depthwise weights to constant memory before launching the kernel.
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

  // Depthwise weight is expected to have shape [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  int total_depthwise = batch * in_channels * out_h * out_w;
  int threads = THREADS_PER_BLOCK;
  int blocks = (total_depthwise + threads - 1) / threads;

  // Before launching the depthwise kernel, copy the weights to constant memory.
  int weight_elements = in_channels * k * k;
  TORCH_CHECK(weight_elements <= MAX_DEPTHWISE_WEIGHT_SIZE, "Depthwise weight size exceeds constant memory limit");
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_depthwise_conv2d_cuda", ([&] {
    size_t weight_size = weight_elements * sizeof(scalar_t);
    if (sizeof(scalar_t) == sizeof(float)) {
      cudaMemcpyToSymbol(depthwise_weight_const_f, depthwise_weight.data_ptr<scalar_t>(), weight_size, 0, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyToSymbol(depthwise_weight_const_d, depthwise_weight.data_ptr<scalar_t>(), weight_size, 0, cudaMemcpyDeviceToDevice);
    }

    optimized_depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(
               (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr<scalar_t>() : nullptr),
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
    printf("Optimized depthwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // Pointwise convolution: weight shape is [out_channels, in_channels, 1, 1]
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
  int total_pointwise = batch * out_channels * out_h * out_w;
  blocks = (total_pointwise + threads - 1) / threads;

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0)
                                     ? pointwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
    pointwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
        depthwise_output.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_weight.data_ptr<scalar_t>()),
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

// Helper function: convert py::object to at::Tensor. If the object is None, returns an undefined tensor.
// If the object has a 'data' attribute (e.g., a torch.nn.Parameter), then that attribute is used.
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

// Wrapper function to handle inputs that may be wrapped in Parameter objects or be None.
// Expected signature: forward(tensor, tensor, tensor, tensor, tensor, int, int, int) -> tensor
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
  m.def("forward", &forward_wrapper, "Optimized CUDA depthwise separable convolution with constant memory");
}
