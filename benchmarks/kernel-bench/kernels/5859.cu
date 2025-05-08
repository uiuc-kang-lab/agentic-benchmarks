#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

__device__ __forceinline__ int compute_start_position(int out_idx, int stride, int padding) {
    return out_idx * stride - padding;
}

__device__ __forceinline__ int compute_pool_bounds(int start, int input_size, int kernel_size, int dilation, bool is_start) {
    if (is_start) {
        return (start < 0) ? ((-start + dilation - 1) / dilation) : 0;
    }
    return min(kernel_size, (input_size - start + dilation - 1) / dilation);
}

template <typename scalar_t>
__global__ void optimized_maxpool3d_kernel(
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

    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int linear_idx = blockIdx.z;

    if (w_out >= output_w || h_out >= output_h) return;

    const int d_out = linear_idx % output_d;
    const int tmp = linear_idx / output_d;
    const int c = tmp % channels;
    const int b = tmp / channels;

    const int d_start = compute_start_position(d_out, stride, padding);
    const int h_start = compute_start_position(h_out, stride, padding);
    const int w_start = compute_start_position(w_out, stride, padding);

    const int k_d_start = compute_pool_bounds(d_start, input_d, kernel_size, dilation, true);
    const int k_d_end = compute_pool_bounds(d_start, input_d, kernel_size, dilation, false);
    const int k_h_start = compute_pool_bounds(h_start, input_h, kernel_size, dilation, true);
    const int k_h_end = compute_pool_bounds(h_start, input_h, kernel_size, dilation, false);
    const int k_w_start = compute_pool_bounds(w_start, input_w, kernel_size, dilation, true);
    const int k_w_end = compute_pool_bounds(w_start, input_w, kernel_size, dilation, false);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    __shared__ scalar_t shared_cache[32][8];

    #pragma unroll
    for (int k_d = k_d_start; k_d < k_d_end; k_d++) {
        const int d_in = d_start + k_d * dilation;
        #pragma unroll
        for (int k_h = k_h_start; k_h < k_h_end; k_h++) {
            const int h_in = h_start + k_h * dilation;
            #pragma unroll
            for (int k_w = k_w_start; k_w < k_w_end; k_w++) {
                const int w_in = w_start + k_w * dilation;
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