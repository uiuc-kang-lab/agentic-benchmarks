#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define TILE_SIZE 32
#define WARP_SIZE 32

template <typename scalar_t>
__device__ __forceinline__ bool is_within_bounds(int coord, int lower, int upper) {
  return coord >= lower && coord < upper;
}

template <typename scalar_t>
__global__ void fused_depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k,
    const int stride,
    const int padding,
    const int dilation) {

  __shared__ scalar_t tile[TILE_SIZE][TILE_SIZE + 1];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ow = blockIdx.x * TILE_SIZE + tx;
  const int oh = blockIdx.y * TILE_SIZE + ty;
  const int n = blockIdx.z / channels;
  const int c = blockIdx.z % channels;

  if (oh < out_h && ow < out_w) {
    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;
    scalar_t sum = (bias != nullptr) ? bias[c] : 0;

    #pragma unroll
    for (int i = 0; i < k; i++) {
      const int ih = ih_base + i * dilation;
      const bool ih_valid = is_within_bounds(ih, 0, in_h);

      #pragma unroll
      for (int j = 0; j < k; j++) {
        const int iw = iw_base + j * dilation;
        if (ih_valid && is_within_bounds(iw, 0, in_w)) {
          sum += input[n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw] *
                 weight[c * k * k + i * k + j];
        }
      }
    }
    output[n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = sum;
  }
}

template <typename scalar_t>
__global__ void optimized_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int h,
    const int w) {

  __shared__ scalar_t weight_shared[TILE_SIZE][TILE_SIZE];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ow = blockIdx.x * TILE_SIZE + tx;
  const int oh = blockIdx.y * TILE_SIZE + ty;
  const int n = blockIdx.z / out_channels;
  const int oc = blockIdx.z % out_channels;

  scalar_t sum = (bias != nullptr) ? bias[oc] : 0;

  for (int ic_base = 0; ic_base < in_channels; ic_base += TILE_SIZE) {
    const int remaining_channels = min(TILE_SIZE, in_channels - ic_base);
    
    if (tx < remaining_channels) {
      weight_shared[ty][tx] = weight[oc * in_channels + ic_base + tx];
    }
    __syncthreads();

    if (oh < h && ow < w) {
      #pragma unroll
      for (int ic_offset = 0; ic_offset < remaining_channels; ic_offset++) {
        const int ic = ic_base + ic_offset;
        sum += input[n * in_channels * h * w + ic * h * w + oh * w + ow] *
               weight_shared[ty][ic_offset];
      }
    }
    __syncthreads();
  }

  if (oh < h && ow < w) {
    output[n * out_channels * h * w + oc * h * w + oh * w + ow] = sum;
  }
}