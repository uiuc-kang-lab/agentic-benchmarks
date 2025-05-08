#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define TILE_DIM 16

// Utility function to get the lower of two values
inline __device__ int min(int a, int b) {
    return a < b ? a : b;
}

// Optimized kernel to distribute workload uniformly with branchless approach
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,   // [batch, channels, in_h, in_w]
    const float* __restrict__ weight,  // [channels, 1, k, k]
    const float* __restrict__ bias,    // [channels] or nullptr
    float* __restrict__ output,        // [batch, channels, out_h, out_w]
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {
  
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int oc = blockIdx.z * blockDim.z + threadIdx.z;

  if (ow < out_w && oh < out_h) {
    int n = oc / channels;
    int c = oc % channels;
    float sum = (bias != nullptr) ? bias[c] : 0;
    
    for (int i = 0; i < k; i++) {
      int ih = oh * stride - padding + i * dilation;
      for (int j = 0; j < k; j++) {
        int iw = ow * stride - padding + j * dilation;
        int valid = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w);
        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
        int weight_idx = c * k * k + i * k + j;
        sum += valid * input[input_idx] * weight[weight_idx];
      }
    }

    int output_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
  }
}

__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ input,   // [batch, in_channels, h, w]
    const float* __restrict__ weight,  // [out_channels, in_channels]
    const float* __restrict__ bias,    // [out_channels] or nullptr
    float* __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h, int w) {

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int oc = blockIdx.z * blockDim.z + threadIdx.z;

  if (ow < w && oh < h) {
    int n = oc / out_channels;
    int co = oc % out_channels;
    float sum = (bias != nullptr) ? bias[co] : 0;
    
    for (int ci = 0; ci < in_channels; ci++) {
      int input_idx = n * in_channels * h * w + ci * h * w + oh * w + ow;
      int weight_idx = co * in_channels + ci;
      sum += input[input_idx] * weight[weight_idx];
    }

    int output_idx = n * out_channels * h * w + co * h * w + oh * w + ow;
    output[output_idx] = sum;
  }
}

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
  int channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());

  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid((out_w + TILE_DIM - 1) / TILE_DIM,
            (out_h + TILE_DIM - 1) / TILE_DIM,
            batch * channels);

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;

  depthwise_conv2d_kernel<<<grid, block>>>(
      x.data_ptr<float>(),
      depthwise_weight.data_ptr<float>(),
      reinterpret_cast<const float*>(depthwise_bias_ptr),
      depthwise_output.data_ptr<float>(),
      batch, channels,
      in_h, in_w,
      out_h, out_w,
      k,
      stride,
      padding,
      dilation);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Depthwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 grid_pw((out_w + TILE_DIM - 1) / TILE_DIM,
               (out_h + TILE_DIM - 1) / TILE_DIM,
               batch * out_channels);

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr() : nullptr;

  pointwise_conv2d_kernel<<<grid_pw, block>>>(
      depthwise_output.data_ptr<float>(),
      pointwise_weight.data_ptr<float>(),
      reinterpret_cast<const float*>(pointwise_bias_ptr),
      output.data_ptr<float>(),
      batch, channels, out_channels, out_h, out_w);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

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
  m.def("forward", &forward_wrapper, "CUDA evenly distributed depthwise separable convolution forward");
}
