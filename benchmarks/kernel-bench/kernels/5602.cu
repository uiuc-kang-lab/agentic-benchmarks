#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

template <typename scalar_t, int KERNEL_SIZE>
__global__ void adaptive_max_pool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride,
    const int padding,
    const int dilation
) {
    if (output_height <= 32 && output_width <= 32) {
        int ow = blockIdx.x * blockDim.x + threadIdx.x;
        int oh = blockIdx.y * blockDim.y + threadIdx.y;
        int bc = blockIdx.z;

        if (ow >= output_width || oh >= output_height) return;

        int b = bc / channels;
        int c = bc % channels;

        int input_batch_stride = channels * input_height * input_width;
        int input_channel_stride = input_height * input_width;
        int base_idx = b * input_batch_stride + c * input_channel_stride;

        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        int ih_start = oh * stride - padding;
        int iw_start = ow * stride - padding;

        if constexpr (KERNEL_SIZE > 0) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                int ih = ih_start + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    int row_idx = base_idx + ih * input_width;
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                        int iw = iw_start + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            max_val = max(max_val, __ldg(&input[row_idx + iw]));
                        }
                    }
                }
            }
        } else {
            for (int kh = 0; kh < -KERNEL_SIZE; ++kh) {
                int ih = ih_start + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    int row_idx = base_idx + ih * input_width;
                    #pragma unroll 4
                    for (int kw = 0; kw < -KERNEL_SIZE; ++kw) {
                        int iw = iw_start + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            max_val = max(max_val, __ldg(&input[row_idx + iw]));
                        }
                    }
                }
            }
        }

        int out_idx = b * (channels * output_height * output_width) + 
                      c * (output_height * output_width) + 
                      oh * output_width + ow;
        output[out_idx] = max_val;
    } else {
        const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (output_idx >= batch_size * channels * output_height * output_width) return;

        const int ow = output_idx % output_width;
        const int oh = (output_idx / output_width) % output_height;
        const int c = (output_idx / (output_width * output_height)) % channels;
        const int b = output_idx / (output_width * output_height * channels);

        const int input_batch_stride = channels * input_height * input_width;
        const int input_channel_stride = input_height * input_width;
        const int base_idx = b * input_batch_stride + c * input_channel_stride;
        
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;

        if constexpr (KERNEL_SIZE > 0) {
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                const int ih = ih_start + kh * dilation;
                if (ih >= 0 && ih < input_height) {
                    const int row_idx = base_idx + ih * input_width;
                    #pragma unroll
                    for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                        const int iw = iw_start + kw * dilation;
                        if (iw >= 0 && iw < input_width) {
                            max_val = max(max_val, __ldg(&input[row_idx + iw]));
                        }
                    }
                }
            }
        }

        output[output_idx] = max_val;
    }
}