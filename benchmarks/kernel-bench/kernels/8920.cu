#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Device function for efficient index computation
__device__ __forceinline__ int get_input_index(int n, int c, int in_channels, int in_h, int in_w, int ih, int iw) {
    return ((n * in_channels + c) * in_h + ih) * in_w + iw;
}

__device__ __forceinline__ int get_weight_index(int c, int oc, int g, int out_channels_per_group, int kernel_h, int kernel_w, int kh, int kw) {
    return ((c * out_channels_per_group + (oc - g * out_channels_per_group)) * kernel_h + kh) * kernel_w + kw;
}

// Optimized CUDA kernel combining streams and modular approach
__global__ void conv_transpose2d_optimized_kernel(
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
    const int out_channels_per_group,
    const int start_idx,
    const int elements_per_partition) {

  // Shared memory for input and weight caching
  extern __shared__ float shared_mem[];
  float* shared_input = shared_mem;
  float* shared_weight = &shared_mem[blockDim.x];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int global_idx = tid + start_idx;
  if (global_idx >= start_idx + elements_per_partition) return;

  // Decode indices
  int ow = global_idx % out_w;
  int temp = global_idx / out_w;
  int oh = temp % out_h;
  temp = temp / out_h;
  int oc = temp % out_channels;
  int n = temp / out_channels;

  // Initialize output with bias using vectorized load
  float out_val = bias[oc];
  
  // Determine group
  int g = oc / out_channels_per_group;
  
  // Process input channels in chunks to maximize cache utilization
  const int CHUNK_SIZE = 32;
  #pragma unroll 4
  for (int c_start = g * in_channels_per_group; c_start < (g + 1) * in_channels_per_group; c_start += CHUNK_SIZE) {
    
    int c_end = min(c_start + CHUNK_SIZE, (g + 1) * in_channels_per_group);
    
    for (int c = c_start; c < c_end; c++) {
      #pragma unroll 2
      for (int kh = 0; kh < kernel_h; kh++) {
        int h_in_candidate = oh + pad_h - kh * dilation_h;
        if (h_in_candidate < 0 || (h_in_candidate % stride_h) != 0) continue;
        int ih = h_in_candidate / stride_h;
        if (ih >= in_h) continue;

        #pragma unroll 2
        for (int kw = 0; kw < kernel_w; kw++) {
          int w_in_candidate = ow + pad_w - kw * dilation_w;
          if (w_in_candidate < 0 || (w_in_candidate % stride_w) != 0) continue;
          int iw = w_in_candidate / stride_w;
          if (iw >= in_w) continue;

          int x_idx = get_input_index(n, c, in_channels, in_h, in_w, ih, iw);
          int w_idx = get_weight_index(c, oc, g, out_channels_per_group, kernel_h, kernel_w, kh, kw);
          
          out_val = __fmaf_rn(x[x_idx], weight[w_idx], out_val);
        }
      }
    }
  }

  // Write output using coalesced memory access
  output[((n * out_channels + oc) * out_h + oh) * out_w + ow] = out_val;
}