#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

__device__ __forceinline__ bool is_valid_input_pos(
    int h_in, int w_in, int H_in, int W_in
) {
    return (h_in >= 0) && (h_in < H_in) && (w_in >= 0) && (w_in < W_in);
}

__global__ void conv2d_cuda_kernel_hybrid(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    bool use_warp_opt
) {
    const int warp_size = 32;
    const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int total_elements = N * C_out * H_out * W_out;

    if (use_warp_opt) {
        const int lane_id = threadIdx.x % warp_size;
        const int warp_id = thread_idx / warp_size;
        const int total_warps = (total_elements + warp_size - 1) / warp_size;
        
        if (warp_id >= total_warps) return;

        const int output_idx = warp_id * warp_size + lane_id;
        if (output_idx >= total_elements) return;

        const int w_out = output_idx % W_out;
        int tmp = output_idx / W_out;
        const int h_out = tmp % H_out;
        tmp = tmp / H_out;
        const int c_out = tmp % C_out;
        const int n = tmp / C_out;

        __shared__ float weight_cache[32][32];

        const int group = c_out / (C_out / groups);
        const int c_in_start = group * (C_in / groups);
        const int c_in_end = c_in_start + (C_in / groups);
        const int C_in_per_group = C_in / groups;

        float value = (bias != nullptr) ? bias[c_out] : 0.0f;

        const int h_in_start = h_out * stride_h - padding_h;
        const int w_in_start = w_out * stride_w - padding_w;

        #pragma unroll
        for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
            const int input_channel_offset = ((n * C_in + c_in) * H_in) * W_in;
            const int weight_channel_offset = ((c_out * C_in_per_group + (c_in - c_in_start)) * K_h) * K_w;

            if (lane_id < K_h * K_w) {
                weight_cache[threadIdx.x / warp_size][lane_id] = weight[weight_channel_offset + lane_id];
            }
            __syncwarp();

            for (int k_h = 0; k_h < K_h; ++k_h) {
                const int h_in = h_in_start + k_h * dilation_h;
                
                if (h_in >= 0 && h_in < H_in) {
                    const int input_h_offset = input_channel_offset + h_in * W_in;

                    for (int k_w = 0; k_w < K_w; ++k_w) {
                        const int w_in = w_in_start + k_w * dilation_w;
                        
                        if (w_in >= 0 && w_in < W_in) {
                            value += input[input_h_offset + w_in] * 
                                    weight_cache[threadIdx.x / warp_size][k_h * K_w + k_w];
                        }
                    }
                }
            }
            __syncwarp();
        }

        if (output_idx < total_elements) {
            output[output_idx] = value;
        }
    } else {
        for (int idx = thread_idx; idx < total_elements; idx += total_threads) {
            const int w_out = idx % W_out;
            int tmp = idx / W_out;
            const int h_out = tmp % H_out;
            tmp = tmp / H_out;
            const int c_out = tmp % C_out;
            const int n = tmp / C_out;

            const int group = c_out / (C_out / groups);
            const int c_in_start = group * (C_in / groups);
            const int c_in_end = c_in_start + (C_in / groups);
            const int C_in_per_group = C_in / groups;

            float value = (bias != nullptr) ? bias[c_out] : 0.0f;

            const int h_in_start = h_out * stride_h - padding_h;
            const int w_in_start = w_out * stride_w - padding_w;

            for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
                const int input_channel_offset = ((n * C_in + c_in) * H_in) * W_in;
                const int weight_channel_offset = ((c_out * C_in_per_group + (c_in - c_in_start)) * K_h) * K_w;

                for (int k_h = 0; k_h < K_h; ++k_h) {
                    const int h_in = h_in_start + k_h * dilation_h;
                    
                    if (h_in >= 0 && h_in < H_in) {
                        const int input_h_offset = input_channel_offset + h_in * W_in;
                        const int weight_h_offset = weight_channel_offset + k_h * K_w;

                        for (int k_w = 0; k_w < K_w; ++k_w) {
                            const int w_in = w_in_start + k_w * dilation_w;
                            
                            if (w_in >= 0 && w_in < W_in) {
                                value += input[input_h_offset + w_in] * 
                                        weight[weight_h_offset + k_w];
                            }
                        }
                    }
                }
            }

            output[idx] = value;
        }
    }
}