#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

template <typename scalar_t>
__global__ void adaptive_max_pool3d_kernel(
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

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    // Decode output coordinates
    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c = (idx / (output_w * output_h * output_d)) % channels;
    const int b = idx / (output_w * output_h * output_d * channels);

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Use shared memory for small kernel sizes (<=5)
    const bool use_shared = (kernel_size <= 5);
    extern __shared__ char shared_mem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared_mem);
    
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int64_t max_index = -1;

    if (use_shared) {
        // Shared memory approach for small kernels
        const int tid = threadIdx.x;
        const int pooling_size = kernel_size * kernel_size * kernel_size;
        
        for (int i = tid; i < pooling_size; i += blockDim.x) {
            const int k_d = i / (kernel_size * kernel_size);
            const int rem = i % (kernel_size * kernel_size);
            const int k_h = rem / kernel_size;
            const int k_w = rem % kernel_size;

            const int d_in = d_start + k_d * dilation;
            const int h_in = h_start + k_h * dilation;
            const int w_in = w_start + k_w * dilation;

            if (d_in >= 0 && d_in < input_d &&
                h_in >= 0 && h_in < input_h &&
                w_in >= 0 && w_in < input_w) {
                const int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                    h_in * input_w + w_in;
                const scalar_t val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_index = input_idx;
                }
            }
        }

        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            const scalar_t other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            const int64_t other_idx = __shfl_down_sync(0xffffffff, max_index, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_index = other_idx;
            }
        }
    } else {
        // Direct computation with loop unrolling for large kernels
        #pragma unroll 4
        for (int k_d = 0; k_d < kernel_size; k_d++) {
            const int d_in = d_start + k_d * dilation;
            if (d_in < 0 || d_in >= input_d) continue;

            #pragma unroll 4
            for (int k_h = 0; k_h < kernel_size; k_h++) {
                const int h_in = h_start + k_h * dilation;
                if (h_in < 0 || h_in >= input_h) continue;

                #pragma unroll 4
                for (int k_w = 0; k_w < kernel_size; k_w++) {
                    const int w_in = w_start + k_w * dilation;
                    if (w_in < 0 || w_in >= input_w) continue;

                    const int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                        h_in * input_w + w_in;
                    const scalar_t val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_index = input_idx;
                    }
                }
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}