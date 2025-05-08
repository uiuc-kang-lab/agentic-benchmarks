#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// Depthwise convolution kernel with optimized memory access
template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
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

  // Align thread indexing to warp size for coalesced memory access
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int total = batch * channels * out_h * out_w;
  
  if (tid >= total)
      return;

  // Decode index ensuring coalesced access within warps
  int ow = tid % out_w;
  int tmp = tid / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int c = tmp % channels;
  int n = tmp / channels;

  scalar_t sum = 0;
  
  // Use __ldg for read-only data
  #pragma unroll
  for (int i = 0; i < k; ++i) {
    #pragma unroll
    for (int j = 0; j < k; ++j) {
      int ih = oh * stride - padding + i * dilation;
      int iw = ow * stride - padding + j * dilation;
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
        int weight_idx = c * k * k + i * k + j;
        sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
      }
    }
  }
  
  if (bias != nullptr) {
    sum += __ldg(&bias[c]);
  }
  
  output[tid] = sum;
}

// Pointwise convolution kernel with optimized memory access
template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int h,
    int w) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int total = batch * out_channels * h * w;
  
  if (tid >= total)
      return;

  // Decode index ensuring coalesced access
  int ow = tid % w;
  int tmp = tid / w;
  int oh = tmp % h;
  tmp = tmp / h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  scalar_t sum = 0;
  
  // Use vectorized loads where possible for better memory bandwidth
  #pragma unroll 4
  for (int ic = 0; ic < in_channels; ++ic) {
    int input_idx = n * in_channels * h * w + ic * h * w + oh * w + ow;
    int weight_idx = oc * in_channels + ic;
    sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
  }
  
  if (bias != nullptr) {
    sum += __ldg(&bias[oc]);
  }
  
  output[tid] = sum;
}

// Rest of the code remains the same as the reference implementation
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
    depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
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
        pointwise_weight.data_ptr<scalar_t>(),
        reinterpret_cast<const scalar_t*>(pointwise_bias_ptr),
        output.data_ptr<scalar_t>(),
        batch,
        in_channels,
        out_channels,
        out_h, out_w);
  }));

  return output;
}

// Helper functions and PYBIND11_MODULE remain the same as reference implementation