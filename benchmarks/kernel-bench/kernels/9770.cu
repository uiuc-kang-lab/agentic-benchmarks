#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define TILE_SIZE 16

__device__ __forceinline__ float compute_depthwise_conv(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int c, int oh, int ow,
    int in_h, int in_w, int channels,
    int kernel_h, int stride, int padding, int dilation) 
{
    float sum = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding + kh * dilation;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
            int weight_idx = c * kernel_h + kh;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    return sum;
}

__global__ void hybrid_depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w, int kernel_h,
    int stride, int padding, int dilation) 
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int tile_idx = blockIdx.x;
    
    const int tiles_per_row = (out_w + TILE_SIZE - 1) / TILE_SIZE;
    const int tile_row = (tile_idx / tiles_per_row) * TILE_SIZE;
    const int tile_col = (tile_idx % tiles_per_row) * TILE_SIZE;
    
    const int bc_idx = blockIdx.y;
    const int b = bc_idx / channels;
    const int c = bc_idx % channels;
    
    if (b >= batch) return;
    
    const int row_offset = warp_id;
    const int oh = tile_row + row_offset;
    
    if (oh < out_h) {
        for (int col_offset = lane_id; col_offset < TILE_SIZE && (tile_col + col_offset) < out_w; col_offset += WARP_SIZE) {
            const int ow = tile_col + col_offset;
            
            float sum = compute_depthwise_conv(
                input, weight, b, c, oh, ow,
                in_h, in_w, channels, kernel_h,
                stride, padding, dilation);
            
            sum += bias[c];
            
            const int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
            output[output_idx] = sum;
        }
    }
}