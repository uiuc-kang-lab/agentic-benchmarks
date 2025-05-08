#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define THREADS_PER_BLOCK 256

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

  const int total = batch * channels * out_h * out_w;
  const int grid_stride = blockDim.x * gridDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
       idx < total; 
       idx += grid_stride) {

    int ow = idx % out_w;
    int tmp = idx / out_w;
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
          int input_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
          int weight_idx = c * k * k + i * k + j;
          sum += input[input_idx] * weight[weight_idx];
        }
      }
    }
    if (bias != nullptr)
      sum += bias[c];
    output[idx] = sum;
  }
}

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

  const int total = batch * out_channels * h * w;
  const int grid_stride = blockDim.x * gridDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += grid_stride) {

    int ow = idx % w;
    int tmp = idx / w;
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
    output[idx] = sum;
  }
}

// Remainder of the code (forward_cuda, toTensor, forward_wrapper, PYBIND11_MODULE)
// remains exactly the same as in the original reference implementation
