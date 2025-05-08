#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Tile sizes for shared memory optimization
#define TILE_D 4
#define TILE_H 4 
#define TILE_W 4

__device__ __forceinline__ int dmin(int a, int b) { 
    return a < b ? a : b; 
}

template <typename scalar_t>
__global__ void max_pool3d_forward_tiled_kernel(
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

    // Shared memory for input tile
    __shared__ scalar_t shared_input[TILE_D][TILE_H][TILE_W];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    // Calculate output indices
    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_w * output_h)) % output_d;
    const int c = (idx / (output_w * output_h * output_d)) % channels;
    const int b = idx / (output_w * output_h * output_d * channels);

    // Calculate input window bounds
    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    // Precompute loop bounds
    const int k_d_start = (d_start < 0) ? ((-d_start + dilation - 1) / dilation) : 0;
    const int k_d_end = dmin(kernel_size, (input_d - d_start + dilation - 1) / dilation);
    const int k_h_start = (h_start < 0) ? ((-h_start + dilation - 1) / dilation) : 0;
    const int k_h_end = dmin(kernel_size, (input_h - h_start + dilation - 1) / dilation);
    const int k_w_start = (w_start < 0) ? ((-w_start + dilation - 1) / dilation) : 0;
    const int k_w_end = dmin(kernel_size, (input_w - w_start + dilation - 1) / dilation);

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    // Process input in tiles
    for (int td = k_d_start; td < k_d_end; td += TILE_D) {
        for (int th = k_h_start; th < k_h_end; th += TILE_H) {
            for (int tw = k_w_start; tw < k_w_end; tw += TILE_W) {
                
                // Load tile into shared memory
                if (tid < TILE_D * TILE_H * TILE_W) {
                    int local_d = tid / (TILE_H * TILE_W);
                    int local_h = (tid / TILE_W) % TILE_H;
                    int local_w = tid % TILE_W;
                    
                    if (td + local_d < k_d_end && 
                        th + local_h < k_h_end && 
                        tw + local_w < k_w_end) {
                        
                        int d_in = d_start + (td + local_d) * dilation;
                        int h_in = h_start + (th + local_h) * dilation;
                        int w_in = w_start + (tw + local_w) * dilation;
                        
                        if (d_in >= 0 && d_in < input_d &&
                            h_in >= 0 && h_in < input_h &&
                            w_in >= 0 && w_in < input_w) {
                            
                            int input_idx = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                            shared_input[local_d][local_h][local_w] = input[input_idx];
                        } else {
                            shared_input[local_d][local_h][local_w] = -std::numeric_limits<scalar_t>::infinity();
                        }
                    }
                }
                __syncthreads();

                // Process tile
                for (int d = 0; d < TILE_D && td + d < k_d_end; d++) {
                    for (int h = 0; h < TILE_H && th + h < k_h_end; h++) {
                        for (int w = 0; w < TILE_W && tw + w < k_w_end; w++) {
                            scalar_t val = shared_input[d][h][w];
                            if (val > max_val) {
                                max_val = val;
                                int d_in = d_start + (td + d) * dilation;
                                int h_in = h_start + (th + h) * dilation;
                                int w_in = w_start + (tw + w) * dilation;
                                max_index = (((b * channels + c) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}