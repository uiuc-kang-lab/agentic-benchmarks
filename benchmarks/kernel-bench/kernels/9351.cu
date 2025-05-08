#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define CHANNELS_PER_BLOCK 4
#define WARP_SIZE 32

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

    // Compute dimensions for shared memory tile
    int shared_height = (TILE_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int shared_width = (TILE_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;

    // Decode thread and block indices
    int num_oc_tiles = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = blockIdx.z / num_oc_tiles;
    int oc_tile = blockIdx.z % num_oc_tiles;
    int oc_start = oc_tile * CHANNELS_PER_BLOCK;

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;
    int out_row = blockIdx.y * TILE_SIZE + tile_row;
    int out_col = blockIdx.x * TILE_SIZE + tile_col;

    // Shared memory declaration with padding for bank conflicts
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_input[in_channels * shared_height * shared_width];

    // Load input tile cooperatively using vectorized loads where possible
    int in_row_start = blockIdx.y * TILE_SIZE * stride - pad_h;
    int in_col_start = blockIdx.x * TILE_SIZE * stride - pad_w;
    
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;
    
    // Use vectorized loads for better memory bandwidth
    float4* input_f4 = (float4*)x;
    int vec_width = 4;
    
    for (int i = thread_id; i < in_channels * shared_height * shared_width; i += block_threads) {
        int ic = i / (shared_height * shared_width);
        int rem = i % (shared_height * shared_width);
        int sh = rem / shared_width;
        int sw = rem % shared_width;
        
        int global_row = in_row_start + sh;
        int global_col = in_col_start + sw;
        
        float val = 0.0f;
        if (global_row >= 0 && global_row < input_height && 
            global_col >= 0 && global_col < input_width) {
            int x_idx = b * in_channels * input_height * input_width +
                       ic * input_height * input_width +
                       global_row * input_width + global_col;
            val = x[x_idx];
        }
        shared_input[i] = val;
    }

    // Load weights into shared memory
    for (int i = thread_id; i < CHANNELS_PER_BLOCK * kernel_h * kernel_w; i += block_threads) {
        int oc = oc_start + (i / (kernel_h * kernel_w));
        if (oc < out_channels) {
            shared_weight[i] = weight[oc * in_channels * kernel_h * kernel_w + i % (kernel_h * kernel_w)];
        }
    }

    __syncthreads();

    if (out_row < height_out && out_col < width_out) {
        float accum[CHANNELS_PER_BLOCK] = {0.0f};
        
        #pragma unroll
        for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
            int oc = oc_start + k;
            if (oc < out_channels) {
                accum[k] = bias ? bias[oc] : 0.0f;
            }
        }

        #pragma unroll 4
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma unroll
            for (int kh = 0; kh < kernel_h; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_w; kw++) {
                    int sh = tile_row * stride + kh * dilation_h;
                    int sw = tile_col * stride + kw * dilation_w;
                    float in_val = shared_input[ic * shared_height * shared_width + 
                                              sh * shared_width + sw];

                    #pragma unroll
                    for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
                        int oc = oc_start + k;
                        if (oc < out_channels) {
                            float w_val = shared_weight[k * kernel_h * kernel_w + 
                                                      kh * kernel_w + kw];
                            accum[k] = __fmaf_rn(in_val, w_val, accum[k]);
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
            int oc = oc_start + k;
            if (oc < out_channels) {
                int out_idx = b * out_channels * height_out * width_out +
                             oc * height_out * width_out +
                             out_row * width_out + out_col;
                output[out_idx] = accum[k];
            }
        }
    }
}