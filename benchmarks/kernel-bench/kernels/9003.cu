#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing convolution when window is fully in bounds
__device__ __forceinline__ float compute_conv1d_inbounds(
    const float* __restrict__ x,
    const float* __restrict__ sweight,
    int b,
    int o,
    int in_channels,
    int in_size,
    int kernel_size,
    int stride,
    int dilation) {
  float sum = 0.0f;
  int start_pos = o * stride;
  
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size + start_pos;
    const float* w_ptr = sweight + ic * kernel_size;
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
      sum += x_ptr[k * dilation] * w_ptr[k];
    }
  }
  return sum;
}

// Device function for computing convolution with boundary checks
__device__ __forceinline__ float compute_conv1d_boundary(
    const float* __restrict__ x,
    const float* __restrict__ sweight,
    int b,
    int o,
    int in_channels,
    int in_size,
    int kernel_size,
    int stride,
    int dilation) {
  float sum = 0.0f;
  int start_pos = o * stride;
  
  for (int ic = 0; ic < in_channels; ++ic) {
    const float* x_ptr = x + b * (in_channels * in_size) + ic * in_size;
    const float* w_ptr = sweight + ic * kernel_size;
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
      int pos = start_pos + k * dilation;
      if (pos < in_size) {
        sum += x_ptr[pos] * w_ptr[k];
      }
    }
  }
  return sum;
}

__global__ void conv1d_hybrid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int kernel_size,
    int out_size,
    int stride,
    int dilation) {
    
    // Each block handles one (batch, output channel) pair
    int b = blockIdx.x;
    int oc = blockIdx.y;
    
    // Load weights into shared memory
    extern __shared__ float sweight[];
    int filter_size = in_channels * kernel_size;
    for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
        sweight[i] = weight[oc * filter_size + i];
    }
    __syncthreads();
    
    float bias_val = (bias != nullptr) ? bias[oc] : 0.0f;
    
    // Calculate boundary position
    int boundary_pos = in_size - (kernel_size - 1) * dilation;
    
    // Each thread handles multiple output positions
    for (int o = threadIdx.x; o < out_size; o += blockDim.x) {
        float sum;
        if (o * stride < boundary_pos) {
            // Use fast path for fully in-bounds computation
            sum = compute_conv1d_inbounds(x, sweight, b, o, in_channels, in_size, 
                                        kernel_size, stride, dilation);
        } else {
            // Use boundary-checking path
            sum = compute_conv1d_boundary(x, sweight, b, o, in_channels, in_size, 
                                        kernel_size, stride, dilation);
        }
        sum += bias_val;
        
        // Write output
        int out_idx = b * (out_channels * out_size) + oc * out_size + o;
        output[out_idx] = sum;
    }
}