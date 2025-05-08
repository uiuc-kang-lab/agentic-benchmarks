#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define CHANNELS_PER_BLOCK 8
#define ELEMENTS_PER_THREAD 4

__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int height_out,
    int width_out,
    int stride,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {

    extern __shared__ float shared_mem[];
    float* shared_weight = shared_mem;
    float* shared_input = &shared_mem[CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int out_x_base = blockIdx.x * BLOCK_SIZE_X * ELEMENTS_PER_THREAD;
    const int out_y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int b = blockIdx.z / ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    const int oc_group = blockIdx.z % ((out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK);
    const int oc_start = oc_group * CHANNELS_PER_BLOCK;

    const int weights_per_thread = (CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w + blockDim.x * blockDim.y - 1) 
                                 / (blockDim.x * blockDim.y);
    #pragma unroll
    for (int i = 0; i < weights_per_thread; i++) {
        const int idx = tid + i * blockDim.x * blockDim.y;
        if (idx < CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w) {
            const int oc_offset = idx / (in_channels * kernel_h * kernel_w);
            const int remainder = idx % (in_channels * kernel_h * kernel_w);
            if (oc_start + oc_offset < out_channels) {
                shared_weight[idx] = __ldg(&weight[(oc_start + oc_offset) * in_channels * kernel_h * kernel_w + remainder]);
            }
        }
    }
    __syncthreads();

    float sums[CHANNELS_PER_BLOCK][ELEMENTS_PER_THREAD] = {0};
    
    #pragma unroll
    for (int oc = 0; oc < CHANNELS_PER_BLOCK; oc++) {
        if (oc_start + oc < out_channels) {
            const float bias_val = bias ? __ldg(&bias[oc_start + oc]) : 0.0f;
            #pragma unroll
            for (int x_offset = 0; x_offset < ELEMENTS_PER_THREAD; x_offset++) {
                sums[oc][x_offset] = bias_val;
            }
        }
    }

    if (out_y < height_out) {
        const int in_y_base = out_y * stride - pad_h;
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                const int in_y = in_y_base + kh * dilation_h;
                if (in_y >= 0 && in_y < input_height) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        #pragma unroll
                        for (int x_offset = 0; x_offset < ELEMENTS_PER_THREAD; x_offset++) {
                            const int out_x = out_x_base + threadIdx.x * ELEMENTS_PER_THREAD + x_offset;
                            if (out_x < width_out) {
                                const int in_x = out_x * stride + kw * dilation_w - pad_w;
                                if (in_x >= 0 && in_x < input_width) {
                                    const float in_val = __ldg(&x[b * in_channels * input_height * input_width +
                                                               ic * input_height * input_width +
                                                               in_y * input_width + in_x]);
                                    
                                    #pragma unroll
                                    for (int oc = 0; oc < CHANNELS_PER_BLOCK; oc++) {
                                        if (oc_start + oc < out_channels) {
                                            const float w_val = shared_weight[oc * in_channels * kernel_h * kernel_w +
                                                                           ic * kernel_h * kernel_w +
                                                                           kh * kernel_w + kw];
                                            sums[oc][x_offset] = __fmaf_rn(in_val, w_val, sums[oc][x_offset]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int x_offset = 0; x_offset < ELEMENTS_PER_THREAD; x_offset++) {
            const int out_x = out_x_base + threadIdx.x * ELEMENTS_PER_THREAD + x_offset;
            if (out_x < width_out) {
                #pragma unroll
                for (int oc = 0; oc < CHANNELS_PER_BLOCK; oc++) {
                    if (oc_start + oc < out_channels) {
                        output[b * out_channels * height_out * width_out +
                               (oc_start + oc) * height_out * width_out +
                               out_y * width_out + out_x] = sums[oc][x_offset];
                    }
                }
            }
        }
    }
}