#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

__device__ inline int dmin(int a, int b) {
    return a < b ? a : b;
}

template <typename scalar_t, int BLOCK_SIZE_X = 32, int BLOCK_SIZE_Y = 8>
__global__ void hybrid_maxpool3d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {

    int w_out, h_out, d_out, c, b;

    if (output_w <= 32 && output_h <= 32) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = batch_size * channels * output_d * output_h * output_w;
        if (idx >= total) return;

        w_out = idx % output_w;
        h_out = (idx / output_w) % output_h;
        d_out = (idx / (output_w * output_h)) % output_d;
        c = (idx / (output_w * output_h * output_d)) % channels;
        b = idx / (output_w * output_h * output_d * channels);
    } else {
        w_out = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
        h_out = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
        int linear_idx = blockIdx.z;

        if (w_out >= output_w || h_out >= output_h) return;

        d_out = linear_idx % output_d;
        int tmp = linear_idx / output_d;
        c = tmp % channels;
        b = tmp / channels;
    }

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    const int k_d_start = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    const int k_d_end = dmin(kernel_size, (input_d - d_start + dilation - 1) / dilation);

    const int k_h_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    const int k_h_end = dmin(kernel_size, (input_h - h_start + dilation - 1) / dilation);

    const int k_w_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    const int k_w_end = dmin(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    #pragma unroll
    for (int kd = k_d_start; kd < k_d_end; kd++) {
        const int d_in = d_start + kd * dilation;
        #pragma unroll
        for (int kh = k_h_start; kh < k_h_end; kh++) {
            const int h_in = h_start + kh * dilation;
            #pragma unroll
            for (int kw = k_w_start; kw < k_w_end; kw++) {
                const int w_in = w_start + kw * dilation;
                const int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                const scalar_t val = __ldg(&input[input_idx]);
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    const int output_idx = (((b * channels + c) * output_d + d_out) * output_h + h_out) * output_w + w_out;
    output[output_idx] = max_val;
    if (indices != nullptr) {
        indices[output_idx] = max_index;
    }
}