#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Shared memory tile dimensions
#define TILE_W 16
#define TILE_H 16 
#define BLOCK_SIZE 256

// Shared memory tiles for input and weights
__shared__ float input_tile[TILE_H][TILE_W];
__shared__ float weight_tile[TILE_H][TILE_W];

__global__ void conv2d_cuda_kernel(
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
    // Block and thread indices
    const int tx = threadIdx.x % TILE_W;
    const int ty = threadIdx.x / TILE_W;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Calculate output positions
    const int n = bz / C_out;
    const int c_out = bz % C_out;
    const int h_out_start = by * TILE_H;
    const int w_out_start = bx * TILE_W;
    
    // Calculate group information
    const int group = c_out / (C_out / groups);
    const int c_in_start = group * (C_in / groups);
    const int c_in_end = c_in_start + (C_in / groups);

    // Initialize output value with bias if available
    float value = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over input channels in the group
    for (int c_in = c_in_start; c_in < c_in_end; c_in += TILE_W) {
        for (int k_tile_h = 0; k_tile_h < K_h; k_tile_h += TILE_H) {
            // Collaborative loading of input and weight tiles
            if (ty < min(TILE_H, K_h - k_tile_h) && tx < min(TILE_W, c_in_end - c_in)) {
                const int k_h = k_tile_h + ty;
                const int c_in_idx = c_in + tx;
                
                if (c_in_idx < c_in_end) {
                    weight_tile[ty][tx] = weight[
                        ((c_out * (C_in / groups) + (c_in_idx - c_in_start)) * K_h + k_h) * K_w
                    ];
                }
            }
            __syncthreads();

            // Process the tiles
            if (h_out_start + ty < H_out && w_out_start + tx < W_out) {
                for (int k_h = 0; k_h < min(TILE_H, K_h - k_tile_h); ++k_h) {
                    for (int k_w = 0; k_w < K_w; ++k_w) {
                        const int h_in = (h_out_start + ty) * stride_h - padding_h + (k_tile_h + k_h) * dilation_h;
                        const int w_in = (w_out_start + tx) * stride_w - padding_w + k_w * dilation_w;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            for (int c_offset = 0; c_offset < min(TILE_W, c_in_end - c_in); ++c_offset) {
                                const int c_in_idx = c_in + c_offset;
                                const float input_val = input[
                                    ((n * C_in + c_in_idx) * H_in + h_in) * W_in + w_in
                                ];
                                value += input_val * weight_tile[k_h][c_offset];
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write output
    if (h_out_start + ty < H_out && w_out_start + tx < W_out) {
        const int output_idx = ((n * C_out + c_out) * H_out + (h_out_start + ty)) * W_out + (w_out_start + tx);
        output[output_idx] = value;
    }
}