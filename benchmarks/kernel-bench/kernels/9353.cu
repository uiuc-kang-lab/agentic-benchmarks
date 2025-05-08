#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Configuration parameters
#define TILE_SIZE 16
#define CHANNELS_PER_BLOCK 4

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

    // Compute dimensions for input tile
    int shared_height = (TILE_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int shared_width = (TILE_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;

    // Thread and block indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // Decode batch and channel indices
    int groups_per_batch = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = bz / groups_per_batch;
    int g = bz % groups_per_batch;
    int oc_start = g * CHANNELS_PER_BLOCK;

    // Output coordinates
    int h_out = by * TILE_SIZE + ty;
    int w_out = bx * TILE_SIZE + tx;

    // Early exit if out of bounds
    if (h_out >= height_out || w_out >= width_out || b >= batch_size) return;

    // Shared memory declarations
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[in_channels * shared_height * shared_width];

    // Initialize output accumulators
    float sums[CHANNELS_PER_BLOCK];
    #pragma unroll
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int global_oc = oc_start + i;
        sums[i] = (global_oc < out_channels && bias != nullptr) ? bias[global_oc] : 0.0f;
    }

    // Cooperative loading of input tile
    int thread_id = ty * blockDim.x + tx;
    int block_threads = blockDim.x * blockDim.y;
    int in_row_start = by * TILE_SIZE * stride - pad_h;
    int in_col_start = bx * TILE_SIZE * stride - pad_w;

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
            val = __ldg(&x[((b * in_channels + ic) * input_height + global_row) * input_width + global_col]);
        }
        shared_input[i] = val;
    }

    // Cooperative loading of weights
    int weight_elements = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;
    for (int i = thread_id; i < weight_elements; i += block_threads) {
        int oc_offset = i / (in_channels * kernel_h * kernel_w);
        int global_oc = oc_start + oc_offset;
        if (global_oc < out_channels) {
            shared_weight[i] = weight[global_oc * in_channels * kernel_h * kernel_w + 
                                    (i % (in_channels * kernel_h * kernel_w))];
        }
    }

    __syncthreads();

    // Compute convolution
    for (int ic = 0; ic < in_channels; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            int sh = ty * stride + kh * dilation_h;
            for (int kw = 0; kw < kernel_w; kw++) {
                int sw = tx * stride + kw * dilation_w;
                
                float in_val = shared_input[ic * shared_height * shared_width + 
                                          sh * shared_width + sw];

                #pragma unroll
                for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
                    int weight_idx = i * (in_channels * kernel_h * kernel_w) +
                                   ic * kernel_h * kernel_w +
                                   kh * kernel_w + kw;
                    sums[i] += in_val * shared_weight[weight_idx];
                }
            }
        }
    }

    // Write results
    for (int i = 0; i < CHANNELS_PER_BLOCK; i++) {
        int global_oc = oc_start + i;
        if (global_oc < out_channels) {
            output[((b * out_channels + global_oc) * height_out + h_out) * width_out + w_out] = sums[i];
        }
    }
}