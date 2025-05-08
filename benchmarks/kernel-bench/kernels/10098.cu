#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define TILE_SIZE 16

// Depthwise convolution kernel with branchless bounds checking to minimize warp divergence.
template <typename scalar_t>
__global__ void branchless_depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, channels, in_h, in_w]
    const scalar_t* __restrict__ weight,  // [channels, 1, k, k]
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
  
  // Each thread is responsible for one output pixel in a given (batch, channel)
  int linear_idx = blockIdx.z;
  int n = linear_idx / channels;
  int c = linear_idx % channels;

  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;

  // Only compute if within output bounds
  if (oh < out_h && ow < out_w) {
    scalar_t sum = (bias != nullptr) ? bias[c] : static_cast<scalar_t>(0);
    
    // Loop over the kernel window
    for (int i = 0; i < k; i++) {
      // Compute the input row index for the kernel element
      int ih = oh * stride - padding + i * dilation;
      
      // Compute branchless indicator for row validity
      // Using unsigned comparison to avoid branching
      int valid_ih = ((unsigned)ih < (unsigned)in_h);
      // If not valid, use 0 as safe index (reads from the first row, which is safe because it exists)
      int safe_ih = valid_ih ? ih : 0;
      
      for (int j = 0; j < k; j++) {
        int iw = ow * stride - padding + j * dilation;
        int valid_iw = ((unsigned)iw < (unsigned)in_w);
        int safe_iw = valid_iw ? iw : 0;
        
        // Overall validity (1 if both valid, 0 otherwise)
        int valid = valid_ih * valid_iw;

        // Compute linear indices with safe indices for out-of-bound accesses
        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + safe_ih * in_w + safe_iw;
        int weight_idx = c * k * k + i * k + j;

        // Multiply by valid mask to nullify contributions when out-of-bound
        sum += valid * input[input_idx] * weight[weight_idx];
      }
    }
    
    int output_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
    output[output_idx] = sum;
  }
}

// Pointwise convolution kernel (1x1 conv) using conventional uniform control flow
template <typename scalar_t>
__global__ void branchless_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [batch, in_channels, h, w]
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, h, w]
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
    
    for (int ic = 0; ic < in_channels; ic++) {
      int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
      int weight_idx = oc * in_channels + ic;
      sum += input[input_idx] * weight[weight_idx];
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
  int channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, channels, out_h, out_w}, x.options());

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
            (out_h + TILE_SIZE - 1) / TILE_SIZE,
            batch * channels);

  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0) ? depthwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "branchless_depthwise_conv2d_cuda", ([&] {
    branchless_depthwise_conv2d_kernel<scalar_t><<<grid, block>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(depthwise_bias_ptr),
        depthwise_output.data_ptr<scalar_t>(),
        batch,
        channels,
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

  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 block_pw(TILE_SIZE, TILE_SIZE);
  dim3 grid_pw((out_w + TILE_SIZE - 1) / TILE_SIZE,
               (out_h + TILE_SIZE - 1) / TILE_SIZE,
               batch * out_channels);

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0) ? pointwise_bias.data_ptr() : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "branchless_pointwise_conv2d_cuda", ([&] {
    branchless_pointwise_conv2d_kernel<scalar_t><<<grid_pw, block_pw>>>(
        depthwise_output.data_ptr<scalar_t>(),
        pointwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        channels,
        out_channels,
        out_h, out_w);
  }));

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Pointwise kernel launch error: %s\n", cudaGetErrorString(err));
  }

  return output;
}

// Helper: converts a py::object to an at::Tensor (supports tensors or objects with a 'data' attribute).
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
  m.def("forward", &forward_wrapper, "CUDA branchless depthwise separable convolution forward");
}
