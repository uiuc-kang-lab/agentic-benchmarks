#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int TILE_SIZE_H = 16;
constexpr int TILE_SIZE_W = 16;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ void load_input_tile(
    const float* __restrict__ input,
    float* shared_input,
    int n, int c_in, int h_start, int w_start,
    int H_in, int W_in, int tile_h, int tile_w,
    int thread_idx
) {
    const int elements_per_thread = (TILE_SIZE_H * TILE_SIZE_W + WARP_SIZE - 1) / WARP_SIZE;
    const int input_offset = ((n * C_in + c_in) * H_in + h_start) * W_in + w_start;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = thread_idx + i * WARP_SIZE;
        if (idx < tile_h * tile_w) {
            int local_h = idx / tile_w;
            int local_w = idx % tile_w;
            if ((h_start + local_h) < H_in && (w_start + local_w) < W_in) {
                shared_input[local_h * tile_w + local_w] = 
                    input[input_offset + local_h * W_in + local_w];
            }
        }
    }
}

__device__ __forceinline__ void load_weight_tile(
    const float* __restrict__ weight,
    float* shared_weight,
    int c_out, int c_in, int K_h, int K_w,
    int C_in_per_group, int thread_idx
) {
    const int elements_per_thread = (K_h * K_w + WARP_SIZE - 1) / WARP_SIZE;
    const int weight_offset = ((c_out * C_in_per_group + c_in) * K_h) * K_w;
    
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = thread_idx + i * WARP_SIZE;
        if (idx < K_h * K_w) {
            shared_weight[idx] = weight[weight_offset + idx];
        }
    }
}

__global__ void conv2d_cuda_kernel_optimized(
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
    int groups
) {
    extern __shared__ float shared_memory[];
    float* shared_input = shared_memory;
    float* shared_weight = shared_memory + TILE_SIZE_H * TILE_SIZE_W;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    
    const int total_warps = (N * C_out * H_out * W_out + WARP_SIZE - 1) / WARP_SIZE;
    if (warp_id >= total_warps) return;

    const int output_idx = warp_id * WARP_SIZE + lane_id;
    if (output_idx >= N * C_out * H_out * W_out) return;

    const int w_out = output_idx % W_out;
    int tmp = output_idx / W_out;
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
        load_input_tile(input, shared_input, n, c_in,
                       h_in_start, w_in_start,
                       H_in, W_in,
                       min(TILE_SIZE_H, H_in - h_in_start),
                       min(TILE_SIZE_W, W_in - w_in_start),
                       lane_id);
        
        load_weight_tile(weight, shared_weight,
                        c_out, c_in - c_in_start,
                        K_h, K_w, C_in_per_group,
                        lane_id);
        
        __syncwarp();

        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            const int h_in = k_h * dilation_h;
            if (h_in >= 0 && h_in < TILE_SIZE_H) {
                #pragma unroll
                for (int k_w = 0; k_w < K_w; ++k_w) {
                    const int w_in = k_w * dilation_w;
                    if (w_in >= 0 && w_in < TILE_SIZE_W) {
                        value += shared_input[h_in * TILE_SIZE_W + w_in] *
                                shared_weight[k_h * K_w + k_w];
                    }
                }
            }
        }
        
        __syncwarp();
    }

    if (output_idx < N * C_out * H_out * W_out) {
        output[output_idx] = value;
    }
}