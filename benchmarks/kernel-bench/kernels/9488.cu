#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <pybind11/pybind11.h>

// Device function for index decoding
__device__ inline void decode_output_index(int index, int out_width, int out_height, int out_channels,
                                    int &w_out, int &h_out, int &o, int &b) {
    w_out = index % out_width;
    int temp = index / out_width;
    h_out = temp % out_height;
    temp /= out_height;
    o = temp % out_channels;
    b = temp / out_channels;
}

__global__ void conv_transpose2d_forward_kernel_hybrid(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation) {
    
    // Shared memory for weight caching
    extern __shared__ float shared_weight[];
    
    // Load weights into shared memory cooperatively
    int tid = threadIdx.x;
    int weight_elements = in_channels * kernel_size * kernel_size;
    int weight_per_thread = (weight_elements + blockDim.x - 1) / blockDim.x;
    
    // Each thread loads multiple weight elements
    for(int i = 0; i < weight_per_thread; i++) {
        int idx = tid + i * blockDim.x;
        if(idx < weight_elements) {
            int c = idx / (kernel_size * kernel_size);
            int remainder = idx % (kernel_size * kernel_size);
            int p = remainder / kernel_size;
            int q = remainder % kernel_size;
            
            // Get output channel for this block
            int o = blockIdx.y;
            shared_weight[idx] = weight[((c * out_channels + o) * kernel_size + p) * kernel_size + q];
        }
    }
    __syncthreads();

    // Process output elements
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int elements_per_channel = out_height * out_width;
    if (index >= elements_per_channel) return;
    
    // Current output channel and batch
    int o = blockIdx.y;
    int b = blockIdx.z;
    
    // Decode position within output channel
    int w_out = index % out_width;
    int h_out = index / out_width;
    
    float out_val = bias[o];
    
    // Compute convolution using shared memory for weights
    for (int c = 0; c < in_channels; ++c) {
        for (int p = 0; p < kernel_size; ++p) {
            int h_unscaled = h_out + padding - p * dilation;
            if (h_unscaled % stride != 0) continue;
            
            int h_in = h_unscaled / stride;
            if (h_in < 0 || h_in >= in_height) continue;
            
            for (int q = 0; q < kernel_size; ++q) {
                int w_unscaled = w_out + padding - q * dilation;
                if (w_unscaled % stride != 0) continue;
                
                int w_in = w_unscaled / stride;
                if (w_in < 0 || w_in >= in_width) continue;
                
                int input_idx = ((b * in_channels + c) * in_height + h_in) * in_width + w_in;
                int weight_idx = c * kernel_size * kernel_size + p * kernel_size + q;
                
                out_val += input[input_idx] * shared_weight[weight_idx];
            }
        }
    }
    
    // Write output
    int output_idx = ((b * out_channels + o) * out_height + h_out) * out_width + w_out;
    output[output_idx] = out_val;
}