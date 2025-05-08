#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Shared memory tile size
#define TILE_SIZE 16

__global__ void optimized_conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

  // Shared memory for input and weight tiles
  __shared__ float s_input[TILE_SIZE][TILE_SIZE];
  __shared__ float s_weight[TILE_SIZE][TILE_SIZE];

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * out_h * out_w;
  if (index >= total) return;

  // Decode flat index into (n, oc, oh, ow)
  int ow = index % out_w;
  int tmp = index / out_w;
  int oh = tmp % out_h;
  tmp = tmp / out_h;
  int oc = tmp % out_channels;
  int n = tmp / out_channels;

  int g = oc / out_channels_per_group;
  float out_val = bias[oc];

  // Process input in tiles
  for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c += TILE_SIZE) {
    for (int kh_base = 0; kh_base < kernel_h; kh_base += TILE_SIZE) {
      
      // Load input tile cooperatively
      if (threadIdx.x < TILE_SIZE) {
        for (int t = 0; t < TILE_SIZE; t++) {
          int curr_c = c + threadIdx.x;
          int curr_kh = kh_base + t;
          
          if (curr_c < (g + 1) * in_channels_per_group && curr_kh < kernel_h) {
            int h_in_candidate = oh + pad_h - curr_kh * dilation_h;
            if (h_in_candidate >= 0 && (h_in_candidate % stride_h) == 0) {
              int ih = h_in_candidate / stride_h;
              if (ih < in_h) {
                s_input[threadIdx.x][t] = x[n * (in_channels * in_h * in_w) + 
                                          curr_c * (in_h * in_w) + 
                                          ih * in_w];
              }
            }
          }
        }
      }

      // Load weight tile cooperatively
      if (threadIdx.x < TILE_SIZE) {
        for (int t = 0; t < TILE_SIZE; t++) {
          int curr_c = c + threadIdx.x;
          int curr_kh = kh_base + t;
          
          if (curr_c < (g + 1) * in_channels_per_group && curr_kh < kernel_h) {
            s_weight[threadIdx.x][t] = weight[curr_c * (out_channels_per_group * kernel_h * kernel_w) +
                                            (oc - g * out_channels_per_group) * (kernel_h * kernel_w) +
                                            curr_kh * kernel_w];
          }
        }
      }
      
      __syncthreads();

      // Compute using tiles
      for (int t_c = 0; t_c < TILE_SIZE && (c + t_c) < (g + 1) * in_channels_per_group; t_c++) {
        for (int t_kh = 0; t_kh < TILE_SIZE && (kh_base + t_kh) < kernel_h; t_kh++) {
          for (int kw = 0; kw < kernel_w; kw++) {
            int w_in_candidate = ow + pad_w - kw * dilation_w;
            if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
            int iw = w_in_candidate / stride_w;
            if (iw >= in_w) continue;

            out_val += s_input[t_c][t_kh] * s_weight[t_c][t_kh];
          }
        }
      }

      __syncthreads();
    }
  }

  output[n * (out_channels * out_h * out_w) +
         oc * (out_h * out_w) +
         oh * out_w + ow] = out_val;
}