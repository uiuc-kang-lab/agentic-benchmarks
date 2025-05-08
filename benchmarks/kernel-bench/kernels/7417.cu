#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Tile sizes for shared memory optimization
#define TILE_SIZE 16
#define WARP_SIZE 32

__global__ void conv_transpose2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int chunkN, int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding,
    int H_out, int W_out) {

    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;

    // Block handles TILE_SIZE x TILE_SIZE region of output
    int out_h_start = blockIdx.y * TILE_SIZE;
    int out_w_start = blockIdx.z * TILE_SIZE;
    int n = blockIdx.x % chunkN;
    int oc_block = blockIdx.x / chunkN;

    // Thread indices within tile
    int th = threadIdx.y;
    int tw = threadIdx.x;

    // Register for accumulating partial sums
    float sum = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < C_in; ++ic) {
        // Determine input region that affects this output tile
        int in_h_start = (out_h_start + padding) / stride;
        int in_w_start = (out_w_start + padding) / stride;
        
        // Load input tile to shared memory
        if (th < TILE_SIZE && tw < TILE_SIZE) {
            int in_h = in_h_start + th;
            int in_w = in_w_start + tw;
            if (in_h < H_in && in_w < W_in) {
                shared_input[th * TILE_SIZE + tw] = 
                    input[n * (C_in * H_in * W_in) + 
                          ic * (H_in * W_in) + 
                          in_h * W_in + in_w];
            }
        }

        // Load weight tile to shared memory
        if (th < K && tw < K) {
            shared_weight[th * K + tw] = 
                weight[ic * (C_out * K * K) + 
                       oc_block * (K * K) + 
                       th * K + tw];
        }
        __syncthreads();

        // Compute partial sums for this input channel
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int out_h = out_h_start + th;
                int out_w = out_w_start + tw;
                
                if (out_h < H_out && out_w < W_out) {
                    int in_h = (out_h + padding - kh);
                    int in_w = (out_w + padding - kw);
                    
                    if (in_h % stride == 0 && in_w % stride == 0) {
                        in_h /= stride;
                        in_w /= stride;
                        if (in_h >= 0 && in_h < H_in && 
                            in_w >= 0 && in_w < W_in) {
                            int in_idx = (in_h - in_h_start) * TILE_SIZE + 
                                       (in_w - in_w_start);
                            sum += shared_input[in_idx] * 
                                  shared_weight[kh * K + kw];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write result
    int out_h = out_h_start + th;
    int out_w = out_w_start + tw;
    if (out_h < H_out && out_w < W_out) {
        int out_idx = n * (C_out * H_out * W_out) +
                     oc_block * (H_out * W_out) +
                     out_h * W_out + out_w;
        output[out_idx] = sum;
    }
}