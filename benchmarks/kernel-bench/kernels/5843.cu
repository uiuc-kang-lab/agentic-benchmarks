#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

__device__ __forceinline__ void compute_output_indices(
    const int idx, const int output_w, const int output_h, const int output_d, const int channels,
    int& w_out, int& h_out, int& d_out, int& c, int& b) {
    w_out = idx % output_w;
    h_out = (idx / output_w) % output_h;
    d_out = (idx / (output_w * output_h)) % output_d;
    c = (idx / (output_w * output_h * output_d)) % channels;
    b = idx / (output_w * output_h * output_d * channels);
}

__device__ __forceinline__ int compute_input_index(
    const int b, const int c, const int d, const int h, const int w,
    const int channels, const int input_d, const int input_h, const int input_w) {
    return (((b * channels + c) * input_d + d) * input_h + h) * input_w + w;
}

template <typename scalar_t, int KERNEL_SIZE=3>
__global__ void max_pool3d_forward_kernel_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int stride,
    const int padding,
    const int dilation) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    __shared__ int shared_dims[5];
    if (threadIdx.x == 0) {
        shared_dims[0] = output_w;
        shared_dims[1] = output_h;
        shared_dims[2] = output_d;
        shared_dims[3] = channels;
        shared_dims[4] = input_w;
    }
    __syncthreads();

    int w_out, h_out, d_out, c, b;
    compute_output_indices(idx, shared_dims[0], shared_dims[1], shared_dims[2], shared_dims[3], 
                         w_out, h_out, d_out, c, b);

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    #pragma unroll
    for (int k_d = 0; k_d < KERNEL_SIZE; k_d++) {
        const int d_in = d_start + k_d * dilation;
        if (d_in < 0 || d_in >= input_d) continue;

        #pragma unroll
        for (int k_h = 0; k_h < KERNEL_SIZE; k_h++) {
            const int h_in = h_start + k_h * dilation;
            if (h_in < 0 || h_in >= input_h) continue;

            #pragma unroll
            for (int k_w = 0; k_w < KERNEL_SIZE; k_w++) {
                const int w_in = w_start + k_w * dilation;
                if (w_in < 0 || w_in >= input_w) continue;

                const int input_idx = compute_input_index(b, c, d_in, h_in, w_in,
                                                        shared_dims[3], input_d, input_h, shared_dims[4]);
                const scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}