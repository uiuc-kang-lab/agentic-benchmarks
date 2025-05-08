#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE = 256>
__global__ void conv_transpose2d_hybrid_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
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
    __shared__ scalar_t shared_input[BLOCK_SIZE];
    __shared__ scalar_t shared_weight[BLOCK_SIZE];

    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (thread_id < total_elements && bias != nullptr) {
        int n = thread_id;
        const int ow = n % out_width;
        n /= out_width;
        const int oh = n % out_height;
        n /= out_height;
        const int oc = n % out_channels;
        output[thread_id] = bias[oc];
    }
    
    __syncthreads();

    for (int idx = thread_id; idx < total_elements; idx += gridDim.x * blockDim.x) {
        int n = idx;
        const int ow = n % out_width;
        n /= out_width;
        const int oh = n % out_height;
        n /= out_height;
        const int oc = n % out_channels;
        n /= out_channels;
        const int b = n;

        const int g = oc / out_channels_per_group;
        const int oc_group = oc % out_channels_per_group;
        const int ic_start = g * in_channels_per_group;

        scalar_t sum = output[idx];

        const int h_start = max(0, (oh + padding) / stride);
        const int h_end = min(in_height, (oh + padding + 1) / stride + 1);
        const int w_start = max(0, (ow + padding) / stride);
        const int w_end = min(in_width, (ow + padding + 1) / stride + 1);

        for (int h_in = h_start; h_in < h_end; h_in++) {
            const int kh = oh + padding - h_in * stride;
            if (kh < 0 || kh >= kernel_h * dilation || kh % dilation != 0) continue;
            
            for (int w_in = w_start; w_in < w_end; w_in++) {
                const int kw = ow + padding - w_in * stride;
                if (kw < 0 || kw >= kernel_w * dilation || kw % dilation != 0) continue;

                for (int ic = threadIdx.x; ic < in_channels_per_group; ic += blockDim.x) {
                    const int input_idx = b * (in_channels * in_height * in_width) +
                                        (ic_start + ic) * (in_height * in_width) +
                                        h_in * in_width + w_in;
                    shared_input[ic] = input[input_idx];
                    
                    const int weight_idx = (ic_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                         oc_group * (kernel_h * kernel_w) +
                                         (kh/dilation) * kernel_w + (kw/dilation);
                    shared_weight[ic] = weight[weight_idx];
                }
                
                __syncthreads();

                for (int ic = 0; ic < in_channels_per_group; ic++) {
                    sum += shared_input[ic] * shared_weight[ic];
                }
                
                __syncthreads();
            }
        }
        
        output[idx] = sum;
    }
}