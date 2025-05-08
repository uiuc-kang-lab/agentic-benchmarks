#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

// Tile sizes for output channels and spatial dimensions
#define TILE_OC 32
#define TILE_SP 8
#define VECTOR_SIZE 4 // Vector loading size for coalesced memory access

__device__ __forceinline__ int gcd(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__device__ __forceinline__ int my_min(int a, int b) {
  return a < b ? a : b;
}

__global__ void conv_transpose2d_kernel_hybrid(
    const float4* __restrict__ x,
    const float4* __restrict__ weight,
    const float* __restrict__ bias,
    float4* __restrict__ output,
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

  // Shared memory for bias and intermediate results
  __shared__ float s_bias[TILE_OC];
  __shared__ float s_temp[TILE_OC][TILE_SP];

  // Calculate tile indices
  int oc_base = blockIdx.x * TILE_OC;
  int sp_base = blockIdx.y * TILE_SP;
  int n = blockIdx.z;

  // Load bias into shared memory
  if (threadIdx.y == 0 && (oc_base + threadIdx.x) < out_channels) {
    s_bias[threadIdx.x] = __ldg(&bias[oc_base + threadIdx.x]);
  }
  __syncthreads();

  // Initialize shared memory temp storage
  s_temp[threadIdx.x][threadIdx.y] = 0.0f;

  // Calculate spatial indices
  int sp_idx = sp_base + threadIdx.y;
  if (sp_idx >= out_h * out_w) return;
  int oh = sp_idx / out_w;
  int ow = sp_idx % out_w;

  // Calculate output channel index
  int oc = oc_base + threadIdx.x;
  if (oc >= out_channels) return;

  // Determine group
  int g = oc / out_channels_per_group;

  // Calculate candidate positions
  int candidate_h = oh + pad_h;
  int candidate_w = ow + pad_w;

  // Calculate valid kernel offsets
  int offset_kh = -1, offset_kw = -1;
  int mod_h = candidate_h % stride_h;
  int mod_w = candidate_w % stride_w;

  #pragma unroll
  for (int k = 0; k < stride_h; k++) {
    if ((k * dilation_h) % stride_h == mod_h) {
      offset_kh = k;
      break;
    }
  }

  #pragma unroll
  for (int k = 0; k < stride_w; k++) {
    if ((k * dilation_w) % stride_w == mod_w) {
      offset_kw = k;
      break;
    }
  }

  int step_kh = stride_h / gcd(stride_h, dilation_h);
  int step_kw = stride_w / gcd(stride_w, dilation_w);
  
  float acc = s_bias[threadIdx.x];

  // Main computation with vectorized memory access
  #pragma unroll
  for (int kh = offset_kh; kh >= 0 && kh < kernel_h; kh += step_kh) {
    int h_in = (candidate_h - kh * dilation_h) / stride_h;
    if (h_in < 0 || h_in >= in_h) continue;

    #pragma unroll
    for (int kw = offset_kw; kw >= 0 && kw < kernel_w; kw += step_kw) {
      int w_in = (candidate_w - kw * dilation_w) / stride_w;
      if (w_in < 0 || w_in >= in_w) continue;

      // Process input channels in vector-size chunks
      for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c += VECTOR_SIZE) {
        float4 x_vec = __ldg((float4*)&x[((n * in_channels + c) * in_h + h_in) * in_w/VECTOR_SIZE + w_in/VECTOR_SIZE]);
        float4 w_vec = __ldg((float4*)&weight[((c * out_channels_per_group + (oc - g * out_channels_per_group)) * kernel_h + kh) * kernel_w/VECTOR_SIZE + kw/VECTOR_SIZE]);
        
        acc += x_vec.x * w_vec.x + x_vec.y * w_vec.y + x_vec.z * w_vec.z + x_vec.w * w_vec.w;
      }
    }
  }

  // Store result in shared memory
  s_temp[threadIdx.x][threadIdx.y] = acc;
  __syncthreads();

  // Vectorized output writing
  if (threadIdx.y == 0 && sp_idx < out_h * out_w) {
    float4 out_vec;
    out_vec.x = s_temp[threadIdx.x][0];
    out_vec.y = s_temp[threadIdx.x][1];
    out_vec.z = s_temp[threadIdx.x][2];
    out_vec.w = s_temp[threadIdx.x][3];
    
    int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w/VECTOR_SIZE + ow/VECTOR_SIZE;
    output[out_idx] = out_vec;
  }
}