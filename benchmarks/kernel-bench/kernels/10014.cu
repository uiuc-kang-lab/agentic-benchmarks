#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32  // Increased tile size for better occupancy
#define BLOCK_SIZE 16
#define MAX_THREADS_PER_BLOCK 256

template<int KERNEL_SIZE>  // Template for compile-time kernel size optimization
__global__ void optimized_depthwise_conv2d_kernel(
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
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group
) {
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int b = bz / out_channels;
    const int c = bz % out_channels;
    const int g = c / channels_per_group;
    const int m = c % channels_per_group;
    
    const int tile_w = TILE_SIZE + (KERNEL_SIZE - 1) * dilation_w;
    const int tile_h = TILE_SIZE + (KERNEL_SIZE - 1) * dilation_h;
    const int in_x_base = bx * TILE_SIZE * stride_w - padding_w;
    const int in_y_base = by * TILE_SIZE * stride_h - padding_h;
    
    #pragma unroll 4
    for (int i = ty; i < tile_h; i += BLOCK_SIZE) {
        const int y = in_y_base + i;
        const bool valid_y = (y >= 0 && y < in_h);
        
        #pragma unroll 4
        for (int j = tx; j < tile_w; j += BLOCK_SIZE * 4) {
            float4 vals;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                const int x = in_x_base + j + k;
                const bool valid = valid_y && (x >= 0 && x < in_w);
                const int input_idx = valid ? 
                    ((b * in_channels + g) * in_h + y) * in_w + x : 0;
                reinterpret_cast<float*>(&vals)[k] = valid ? 
                    input[input_idx] : 0.0f;
            }
            
            if (j + 3 < tile_w) {
                *reinterpret_cast<float4*>(&shared_input[i * tile_w + j]) = vals;
            } else {
                #pragma unroll
                for (int k = 0; k < 4 && (j + k) < tile_w; k++) {
                    shared_input[i * tile_w + j + k] = 
                        reinterpret_cast<float*>(&vals)[k];
                }
            }
        }
    }
    
    __syncthreads();
    
    const int out_x = bx * TILE_SIZE + tx;
    const int out_y = by * TILE_SIZE + ty;
    
    if (out_x < out_w && out_y < out_h) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int sh_y = ty * stride_h + kh * dilation_h;
                const int sh_x = tx * stride_w + kw * dilation_w;
                
                sum += shared_input[sh_y * tile_w + sh_x] * 
                       weight[((g * channels_per_group + m) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw];
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[((b * out_channels + c) * out_h + out_y) * out_w + out_x] = sum;
    }
}