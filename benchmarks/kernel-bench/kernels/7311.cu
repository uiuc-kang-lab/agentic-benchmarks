#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Shared memory tile sizes
#define TILE_SIZE_H 8
#define TILE_SIZE_W 8 

__device__ __forceinline__ void calculate_indices(
    int output_idx, int W_out, int H_out, int C_out,
    int& w_out, int& h_out, int& c_out, int& n
) {
    w_out = output_idx % W_out;
    int tmp = output_idx / W_out;
    h_out = tmp % H_out;
    tmp = tmp / H_out;
    c_out = tmp % C_out;
    n = tmp / C_out;
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
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + (TILE_SIZE_H + K_h - 1) * (TILE_SIZE_W + K_w - 1);

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    
    const int total_warps = (N * C_out * H_out * W_out + warp_size - 1) / warp_size;
    if (warp_id >= total_warps) return;

    const int output_idx = warp_id * warp_size + lane_id;
    if (output_idx >= N * C_out * H_out * W_out) return;

    int w_out, h_out, c_out, n;
    calculate_indices(output_idx, W_out, H_out, C_out, w_out, h_out, c_out, n);

    // Calculate group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);
    const int C_in_per_group = C_in / groups;

    // Initialize accumulator
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Calculate tile boundaries
    const int tile_h_start = (h_out / TILE_SIZE_H) * TILE_SIZE_H;
    const int tile_w_start = (w_out / TILE_SIZE_W) * TILE_SIZE_W;

    // Load input tile into shared memory
    const int h_in_base = tile_h_start * stride_h - padding_h;
    const int w_in_base = tile_w_start * stride_w - padding_w;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        // Collaborative loading of input tile
        for (int i = threadIdx.x; i < (TILE_SIZE_H + K_h - 1) * (TILE_SIZE_W + K_w - 1); i += blockDim.x) {
            int tile_h = i / (TILE_SIZE_W + K_w - 1);
            int tile_w = i % (TILE_SIZE_W + K_w - 1);
            int h_in = h_in_base + tile_h;
            int w_in = w_in_base + tile_w;
            
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                shared_input[tile_h * (TILE_SIZE_W + K_w - 1) + tile_w] = 
                    input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
            } else {
                shared_input[tile_h * (TILE_SIZE_W + K_w - 1) + tile_w] = 0.0f;
            }
        }

        // Load weight tile into shared memory
        for (int i = threadIdx.x; i < K_h * K_w; i += blockDim.x) {
            int k_h = i / K_w;
            int k_w = i % K_w;
            shared_weight[k_h * K_w + k_w] = 
                weight[((c_out * C_in_per_group + (c_in - c_in_start)) * K_h + k_h) * K_w + k_w];
        }

        __syncthreads();

        // Compute convolution using shared memory
        const int h_local = h_out - tile_h_start;
        const int w_local = w_out - tile_w_start;

        #pragma unroll
        for (int k_h = 0; k_h < K_h; ++k_h) {
            #pragma unroll
            for (int k_w = 0; k_w < K_w; ++k_w) {
                const int h_offset = h_local * stride_h + k_h * dilation_h;
                const int w_offset = w_local * stride_w + k_w * dilation_w;
                
                value += shared_input[h_offset * (TILE_SIZE_W + K_w - 1) + w_offset] * 
                         shared_weight[k_h * K_w + k_w];
            }
        }

        __syncthreads();
    }

    output[output_idx] = value;
}