#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256
#define BLOCK_SIZE 16 // Assuming kernel size and input dimensions are compatible.

// Depthwise convolution kernel using shared memory optimization.
template <typename scalar_t>
__global__ void depthwise_conv2d_kernel_shared(
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

  extern __shared__ scalar_t shared_input[];

  int oh = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int ow = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int c = blockIdx.z;
  int n = blockIdx.w;

  if (oh >= out_h || ow >= out_w) return;

  int shared_h = BLOCK_SIZE + k - 1;
  int shared_w = BLOCK_SIZE + k - 1;

  // Load input data to shared memory
  for (int i = threadIdx.y; i < shared_h; i += BLOCK_SIZE)
    for (int j = threadIdx.x; j < shared_w; j += BLOCK_SIZE) {
      int ih = oh * stride - padding + i;
      int iw = ow * stride - padding + j;
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        shared_input[i * shared_w + j] = input[n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
      } else {
        shared_input[i * shared_w + j] = 0;
      }
    }

  __syncthreads();

  scalar_t sum = 0;
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      sum += shared_input[(threadIdx.y + i) * shared_w + (threadIdx.x + j)] * weight[c * k * k + i * k + j];
    }
  }

  if (bias != nullptr)
    sum += bias[c];

  output[n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = sum;
}

// Main forward function with shared memory kernel execution.
torch::Tensor forward_cuda_shared(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

  // Common checks and preparations
  int batch = x.size(0);
  int in_channels = x.size(1);
  int in_h = x.size(2);
  int in_w = x.size(3);

  int k = depthwise_weight.size(2);
  int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
  int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

  auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

  // Define grid and block dimensions
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE, in_channels, batch);
  size_t shared_memory_size = (BLOCK_SIZE + k - 1) * (BLOCK_SIZE + k - 1) * sizeof(scalar_t);

  // Launch optimized depthwise kernel with shared memory
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_shared_cuda", ([&] {
    depthwise_conv2d_kernel_shared<scalar_t><<<grid, block, shared_memory_size>>>(
        x.data_ptr<scalar_t>(),
        depthwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(depthwise_bias.data_ptr()),
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
    printf("Depthwise shared kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // Pointwise convolution remains unchanged as shared memory benefits are limited.
  int out_channels = pointwise_weight.size(0);
  auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
  int total_pointwise = batch * out_channels * out_h * out_w;
  dim3 pointwise_grid((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE, out_channels, batch);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
    pointwise_conv2d_kernel<scalar_t><<<pointwise_grid, block>>>(
        depthwise_output.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_weight.data_ptr<scalar_t>()),
        reinterpret_cast<const scalar_t*>(pointwise_bias.data_ptr()),
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

// Wrapper for shared memory optimizations.
at::Tensor forward_wrapper_shared(py::object x_obj,
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

  return forward_cuda_shared(x, depthwise_weight, pointwise_weight,
                             depthwise_bias, pointwise_bias,
                             stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_wrapper_shared, "CUDA depthwise separable convolution forward with shared memory");
}