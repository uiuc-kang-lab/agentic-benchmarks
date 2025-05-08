#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


__device__ scalar_t compute_conv_value_shared(const scalar_t* input_shared,
    const scalar_t* weight_shared,
    const int oh,
    const int ow,
    const int in_channels_per_group,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int dilation,
    const int out_channels_per_group
) {
    scalar_t val = 0;
    
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int h_in = oh - kh * dilation;
            int w_in = ow - kw * dilation;

            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                scalar_t x_val = input_shared[(ic) * in_height + h_in * in_width + w_in];
                scalar_t w_val = weight_shared[(ic) * out_channels_per_group * kernel_h * kernel_w + kh * kernel_w + kw];

                val += x_val * w_val;
            }
        }
    }
    return val;
}

// device kernel using shared memory and avoiding unnecessary syncthreads

template <typename scalar_t>
__global__ void shared_memory_conv_transpose2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    __shared__ scalar_t input_shared[...]; // shared memory allocation
    __shared__ scalar_t weight_shared[...];  // shared memory allocation

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int b, oc, oh, ow;
    calculate_indices<scalar_t>(idx, out_width, out_height, out_channels, b, oc, oh, ow);
    if (b >= batch_size) return;

    const int out_channels_per_group = out_channels / groups;
    
    // Load input and weight into shared memory
    // Load only necessary via using calculate_infex and conditional clauses -> i.e conditional base + threadId

    // Ensure input and weight blocks are loaded across multiple threads, as needed
    // Implementing !(__syncthreads()) when not necessary

    __syncthreads() 
    scalar_t val = compute_conv_val_shared(input_shared,weight_shared,...) // only when necessary on synchronisation in-shared memory reads

    if (bias != nullptr) {
        val += __ldg(&bias[oc]);
    }

    output[idx] = val;
}

// Forward and pybind11 method code similar to previous versions

