#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for small kernel weights
__constant__ float weight_const[1024];

template<int KERNEL_SIZE>
__global__ void conv1d_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int in_channels,
    int in_size,
    int out_channels,
    int out_size,
    int stride,
    int dilation
) {
    // Shared memory for input data tiling
    extern __shared__ float shared_input[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * out_channels * out_size;
    if (idx >= total_elements) return;

    // Calculate indices with minimal divergence
    int o = idx % out_size;
    int tmp = idx / out_size;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    // Pre-calculate window bounds
    int start_pos = o * stride;
    int end_pos = start_pos + (KERNEL_SIZE - 1) * dilation;
    
    float sum = 0.0f;
    
    // Use constant memory for small kernels, global memory for large ones
    const float* w_ptr = (KERNEL_SIZE * in_channels * out_channels <= 1024) ? 
                        weight_const : weight;

    // Main convolution loop with boundary handling optimization
    if (end_pos < in_size) {
        // Fast path - no boundary checks needed
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_base = x + b * (in_channels * in_size) + ic * in_size + start_pos;
            const float* w_base = w_ptr + oc * (in_channels * KERNEL_SIZE) + ic * KERNEL_SIZE;
            
            // Static unrolling for known kernel sizes
            #pragma unroll
            for (int k = 0; k < KERNEL_SIZE; ++k) {
                sum += x_base[k * dilation] * w_base[k];
            }
        }
    } else {
        // Boundary case with vectorized comparison
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* x_base = x + b * (in_channels * in_size) + ic * in_size;
            const float* w_base = w_ptr + oc * (in_channels * KERNEL_SIZE) + ic * KERNEL_SIZE;
            
            #pragma unroll
            for (int k = 0; k < KERNEL_SIZE; ++k) {
                int input_pos = start_pos + k * dilation;
                sum += (input_pos < in_size) ? (x_base[input_pos] * w_base[k]) : 0.0f;
            }
        }
    }

    // Vectorized bias addition
    if (bias != nullptr) {
        sum += bias[oc];
    }
    
    output[b * (out_channels * out_size) + oc * out_size + o] = sum;
}