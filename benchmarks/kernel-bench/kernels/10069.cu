#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256

// Depthwise convolution kernel with optimized workload distribution
// Performs a depthwise separable 2D convolution on the input with correct boundary handling.
// Implements workload balancing to efficiently use GPU resources.
template <typename scalar_t>
__global__ void depthwise_conv2d_kernel_optimized(
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

  // Thread id calculation for execution in a balanced manner
  int batch_index = blockIdx.x;
  int channel_index = threadIdx.y;
  int output_xy = threadIdx.x + blockDim.x * blockIdx.y;

  // Bounds check
  if (channel_index >= channels || output_xy >= out_h * out_w) return;

  // Calculate the spatial index (oh, ow)
  int ow = output_xy % out_w;
  int oh = output_xy / out_w;

  scalar_t sum = 0;
  // Loop over the kernel spatial dimensions
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      int ih = oh * stride - padding + i * dilation;
      int iw = ow * stride - padding + j * dilation;
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        int input_idx = batch_index * channels * in_h * in_w + channel_index * in_h * in_w + ih * in_w + iw;
        int weight_idx = channel_index * k * k + i * k + j;
        sum += input[input_idx] * weight[weight_idx];
      }
    }
  }
  if (bias != nullptr)
    sum += bias[channel_index];

  int output_idx = batch_index * channels * out_h * out_w + channel_index * out_h * out_w + oh * out_w + ow;
  output[output_idx] = sum;
}

// Pointwise convolution kernel with optimized workload distribution
// Efficient utilization of blocks and threads for a balanced workload.
template <typename scalar_t>
__global__ void pointwise_conv2d_kernel_optimized(
    const scalar_t* __restrict__ input,   // [batch, in_channels, h, w] (output of depthwise)
    const scalar_t* __restrict__ weight,  // [out_channels, in_channels]
    const scalar_t* __restrict__ bias,    // [out_channels] or nullptr
    scalar_t* __restrict__ output,        // [batch, out_channels, h, w]
    int batch,
    int in_channels,
    int out_channels,
    int h,
    int w) {

  // Using 2D blocks and grids to improve performance
  int batch_index = blockIdx.x;
  int output_channel = threadIdx.y;
  int output_xy = threadIdx.x + blockDim.x * blockIdx.y;

  // Bounds check
  if (output_channel >= out_channels || output_xy >= h * w) return;

  // Calculate the spatial index (oh, ow)
  int ow = output_xy % w;
  int oh = output_xy / w;

  scalar_t sum = 0;
  for (int ic = 0; ic < in_channels; ++ic) {
    int input_idx = batch_index * in_channels * h * w + ic * h * w + oh * w + ow;
    int weight_idx = output_channel * in_channels + ic;
    sum += input[input_idx] * weight[weight_idx];
  }
  if (bias != nullptr)
    sum += bias[output_channel];

  int output_idx = batch_index * out_channels * h * w + output_channel * h * w + oh * w + ow;
  output[output_idx] = sum;
}

// The core CUDA forward function with optimized workload balancing.
torch::Tensor forward_cuda_optimized(
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

  // Optimized block dimensions
  dim3 block_size(THREADS_PER_BLOCK, in_channels);
  dim3 num_blocks(batch, (in_h * in_w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

  // Depthwise weight expected shape: [in_channels, 1, k, k]
  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  // If bias tensors are not provided (or empty), pass nullptr to the kernels.
  const void* depthwise_bias_ptr = (depthwise_bias.defined() && depthwise_bias.numel() > 0)
                                     ? depthwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
    depthwise_conv2d_kernel_optimized<scalar_t><<<num_blocks, block_size>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
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

  // Pointwise convolution: weight shape is [out_channels, in_channels, 1, 1].
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

  dim3 pointwise_block_size(THREADS_PER_BLOCK, out_channels);
  dim3 pointwise_num_blocks(batch, (out_h * out_w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

  const void* pointwise_bias_ptr = (pointwise_bias.defined() && pointwise_bias.numel() > 0)
                                     ? pointwise_bias.data_ptr()
                                     : nullptr;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
    pointwise_conv2d_kernel_optimized<scalar_t><<<pointwise_num_blocks, pointwise_block_size>>>(
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

// Helper: convert an input py::object to an at::Tensor.
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

// Expected signature: forward(tensor, tensor, tensor, tensor, tensor, int, int, int) → tensor
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

  return forward_cuda_optimized(x, depthwise_weight, pointwise_weight,
                      depthwise_bias, pointwise_bias,
                      stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper, "CUDA optimized depthwise separable convolution forward");
}
