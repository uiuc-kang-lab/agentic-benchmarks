#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define optimization parameters
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

    // Compute shared memory dimensions
    int shared_height = (TILE_SIZE - 1) * stride + (kernel_h - 1) * dilation_h + 1;
    int shared_width = (TILE_SIZE - 1) * stride + (kernel_w - 1) * dilation_w + 1;

    // Decode block and thread indices
    int num_oc_tiles = (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    int b = blockIdx.z / num_oc_tiles;
    int oc_tile = blockIdx.z % num_oc_tiles;
    int oc_start = oc_tile * CHANNELS_PER_BLOCK;

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;
    int out_row = blockIdx.y * TILE_SIZE + tile_row;
    int out_col = blockIdx.x * TILE_SIZE + tile_col;

    // Declare shared memory for both input and weights
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = &shared_mem[in_channels * shared_height * shared_width];

    // Thread identifiers for cooperative loading
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // Load weights into shared memory (only once per block)
    int weight_elements = CHANNELS_PER_BLOCK * in_channels * kernel_h * kernel_w;
    for (int i = thread_id; i < weight_elements; i += block_threads) {
        int w_oc = i / (in_channels * kernel_h * kernel_w);
        int rem = i % (in_channels * kernel_h * kernel_w);
        int global_oc = oc_start + w_oc;
        shared_weight[i] = (global_oc < out_channels) ? 
            weight[global_oc * in_channels * kernel_h * kernel_w + rem] : 0.0f;
    }

    // Load input tile into shared memory
    int in_row_start = blockIdx.y * TILE_SIZE * stride - pad_h;
    int in_col_start = blockIdx.x * TILE_SIZE * stride - pad_w;
    int input_elements = in_channels * shared_height * shared_width;

    for (int i = thread_id; i < input_elements; i += block_threads) {
        int ic = i / (shared_height * shared_width);
        int rem = i % (shared_height * shared_width);
        int sh = rem / shared_width;
        int sw = rem % shared_width;
        
        int global_row = in_row_start + sh;
        int global_col = in_col_start + sw;
        
        float val = 0.0f;
        if (global_row >= 0 && global_row < input_height && 
            global_col >= 0 && global_col < input_width) {
            val = __ldg(&x[b * in_channels * input_height * input_width +
                          ic * input_height * input_width +
                          global_row * input_width + global_col]);
        }
        shared_input[i] = val;
    }

    __syncthreads();

    // Compute output only if within bounds
    if (out_row < height_out && out_col < width_out) {
        float accum[CHANNELS_PER_BLOCK] = {0.0f};
        
        // Initialize with bias if present
        #pragma unroll
        for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
            int oc = oc_start + k;
            if (oc < out_channels) {
                accum[k] = (bias != nullptr) ? bias[oc] : 0.0f;
            }
        }

        // Main computation loop with improved memory access pattern
        for (int ic = 0; ic < in_channels; ic++) {
            #pragma unroll
            for (int kh = 0; kh < kernel_h; kh++) {
                int sh = tile_row * stride + kh * dilation_h;
                #pragma unroll
                for (int kw = 0; kw < kernel_w; kw++) {
                    int sw = tile_col * stride + kw * dilation_w;
                    float in_val = shared_input[ic * (shared_height * shared_width) + 
                                              sh * shared_width + sw];
                    
                    #pragma unroll
                    for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
                        int weight_idx = k * (in_channels * kernel_h * kernel_w) +
                                       ic * (kernel_h * kernel_w) +
                                       kh * kernel_w + kw;
                        accum[k] += in_val * shared_weight[weight_idx];
                    }
                }
            }
        }

        // Write results to global memory
        #pragma unroll
        for (int k = 0; k < CHANNELS_PER_BLOCK; k++) {
            int oc = oc_start + k;
            if (oc < out_channels) {
                output[b * out_channels * height_out * width_out +
                      oc * height_out * width_out +
                      out_row * width_out + out_col] = accum[k];
            }
        }
    }
}