#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define SMALL_KERNEL_THRESHOLD 5

// 2D block kernel for small convolution kernels
__global__ void depthwise_conv2d_small_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
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
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group
) {
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int b = blockIdx.z / out_channels;
    const int c_out = blockIdx.z % out_channels;

    if (x >= out_w || y >= out_h || b >= batch_size) return;

    const int g = c_out / channels_per_group;
    const int m = c_out % channels_per_group;

    float sum = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_in = y * stride_h - padding_h + kh * dilation_h;
        #pragma unroll
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int w_in = x * stride_w - padding_w + kw * dilation_w;
            
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                const int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                const int weight_idx = ((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[((b * out_channels + c_out) * out_h + y) * out_w + x] = sum;
}

// Warp-based kernel for large convolution kernels
__global__ void depthwise_conv2d_large_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    const int warps_per_block = blockDim.x / WARP_SIZE;
    int global_warp_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    int lane = threadIdx.x % WARP_SIZE;
    int total_outputs = batch_size * out_channels * out_h * out_w;

    for (int out_idx = global_warp_id; out_idx < total_outputs; out_idx += gridDim.x * warps_per_block) {
        int tmp = out_idx;
        int w_out = tmp % out_w;
        tmp /= out_w;
        int h_out = tmp % out_h;
        tmp /= out_h;
        int c_out = tmp % out_channels;
        int b = tmp / out_channels;

        int g = c_out / channels_per_group;
        float sum = 0.0f;

        for (int k = lane; k < kernel_h * kernel_w; k += WARP_SIZE) {
            int kh = k / kernel_w;
            int kw = k % kernel_w;
            int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            int w_in = w_out * stride_w - padding_w + kw * dilation_w;
            
            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                int input_idx = ((b * in_channels + g) * in_h + h_in) * in_w + w_in;
                int weight_idx = ((g * channels_per_group + (c_out % channels_per_group)) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }

        unsigned int mask = 0xffffffff;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        if (lane == 0) {
            if (bias != nullptr)
                sum += bias[c_out];
            output[out_idx] = sum;
        }
    }
}