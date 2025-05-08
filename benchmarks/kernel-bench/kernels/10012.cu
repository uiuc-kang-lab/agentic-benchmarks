#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define BLOCK_SIZE 16
#define MAX_THREADS_PER_BLOCK 256

template<int KERNEL_SIZE>
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
    // Use built-in float4 type from CUDA
    
    const int n = blockIdx.z;
    const int b = n / out_channels;
    const int c = n % out_channels;
    const int g = c / channels_per_group;
    const int m = c % channels_per_group;
    
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_input + (TILE_SIZE + KERNEL_SIZE - 1) * (TILE_SIZE + KERNEL_SIZE - 1);
    
    const int tile_out_y = blockIdx.y * TILE_SIZE;
    const int tile_out_x = blockIdx.x * TILE_SIZE;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    if (tid < KERNEL_SIZE * KERNEL_SIZE) {
        shared_weight[tid] = weight[((g * channels_per_group + m) * KERNEL_SIZE + tid / KERNEL_SIZE) * KERNEL_SIZE + tid % KERNEL_SIZE];
    }
    
    const int in_tile_h = TILE_SIZE + KERNEL_SIZE - 1;
    const int in_tile_w = TILE_SIZE + KERNEL_SIZE - 1;
    
    const int in_tile_start_y = tile_out_y * stride_h - padding_h;
    const int in_tile_start_x = tile_out_x * stride_w - padding_w;
    
    #pragma unroll
    for (int i = tid; i < in_tile_h * in_tile_w; i += blockDim.x * blockDim.y) {
        const int y = in_tile_start_y + (i / in_tile_w);
        const int x = in_tile_start_x + (i % in_tile_w);
        
        float val = 0.0f;
        if (y >= 0 && y < in_h && x >= 0 && x < in_w) {
            val = input[((b * in_channels + g) * in_h + y) * in_w + x];
        }
        shared_input[i] = val;
    }
    
    __syncthreads();
    
    const int out_y = tile_out_y + ty;
    const int out_x = tile_out_x + tx;
    
    if (out_y < out_h && out_x < out_w) {
        float sum = 0.0f;
        
        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                const int sh_y = ty * stride_h + kh * dilation_h;
                const int sh_x = tx * stride_w + kw * dilation_w;
                
                sum += shared_input[sh_y * in_tile_w + sh_x] * 
                       shared_weight[kh * KERNEL_SIZE + kw];
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[((b * out_channels + c) * out_h + out_y) * out_w + out_x] = sum;
    }
}