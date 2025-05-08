#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tunable parameters
#define TILE_D 4
#define TILE_H 4 
#define TILE_W 4
#define CH_TILE 8
#define THREADS_PER_BLOCK 256

__global__ void conv3d_hybrid_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height, 
    const int in_width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int dilation
) {
    // Shared memory for input tile and weights
    extern __shared__ float shared_mem[];
    float* smem_input = shared_mem;
    float* smem_weight = shared_mem + CH_TILE * TILE_D * TILE_H * TILE_W;
    
    // Calculate position
    const int tid = threadIdx.x;
    const int oc = blockIdx.x;
    const int batch_id = blockIdx.y;
    
    // Calculate tile position
    const int tile_idx = blockIdx.z;
    const int tiles_per_slice = (out_height/TILE_H) * (out_width/TILE_W);
    const int od = (tile_idx / tiles_per_slice) * TILE_D;
    const int remaining = tile_idx % tiles_per_slice;
    const int oh = (remaining / (out_width/TILE_W)) * TILE_H;
    const int ow = (remaining % (out_width/TILE_W)) * TILE_W;

    // Process tiles of input channels
    float sum[TILE_D][TILE_H][TILE_W] = {0};
    
    for(int ic_base = 0; ic_base < in_channels; ic_base += CH_TILE) {
        const int curr_ch_tile = min(CH_TILE, in_channels - ic_base);
        
        // Collaborative loading of input tile
        for(int i = tid; i < curr_ch_tile * TILE_D * TILE_H * TILE_W; i += THREADS_PER_BLOCK) {
            int ch = i / (TILE_D * TILE_H * TILE_W);
            int pos = i % (TILE_D * TILE_H * TILE_W);
            int d = pos / (TILE_H * TILE_W);
            int h = (pos % (TILE_H * TILE_W)) / TILE_W;
            int w = pos % TILE_W;
            
            int id = od * stride + d - padding;
            int ih = oh * stride + h - padding;
            int iw = ow * stride + w - padding;
            
            float val = 0.0f;
            if(id >= 0 && id < in_depth && ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                val = input[((batch_id * in_channels + (ic_base + ch)) * in_depth + id) * 
                           in_height * in_width + ih * in_width + iw];
            }
            smem_input[ch * TILE_D * TILE_H * TILE_W + pos] = val;
        }
        
        // Collaborative loading of weights
        for(int i = tid; i < curr_ch_tile * kernel_d * kernel_h * kernel_w; i += THREADS_PER_BLOCK) {
            int ch = i / (kernel_d * kernel_h * kernel_w);
            int kidx = i % (kernel_d * kernel_h * kernel_w);
            smem_weight[i] = weight[((oc * in_channels + (ic_base + ch)) * kernel_d * kernel_h * kernel_w) + kidx];
        }
        
        __syncthreads();
        
        // Compute convolution for this tile
        for(int td = 0; td < TILE_D && (od + td) < out_depth; td++) {
            for(int th = 0; th < TILE_H && (oh + th) < out_height; th++) {
                for(int tw = 0; tw < TILE_W && (ow + tw) < out_width; tw++) {
                    float local_sum = 0.0f;
                    
                    for(int ch = 0; ch < curr_ch_tile; ch++) {
                        for(int kd = 0; kd < kernel_d; kd++) {
                            for(int kh = 0; kh < kernel_h; kh++) {
                                for(int kw = 0; kw < kernel_w; kw++) {
                                    int d_idx = td * stride + kd * dilation;
                                    int h_idx = th * stride + kh * dilation;
                                    int w_idx = tw * stride + kw * dilation;
                                    
                                    float in_val = smem_input[ch * TILE_D * TILE_H * TILE_W + 
                                                             d_idx * TILE_H * TILE_W +
                                                             h_idx * TILE_W + w_idx];
                                    float w_val = smem_weight[ch * kernel_d * kernel_h * kernel_w +
                                                            kd * kernel_h * kernel_w +
                                                            kh * kernel_w + kw];
                                    local_sum += in_val * w_val;
                                }
                            }
                        }
                    }
                    sum[td][th][tw] += local_sum;
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    for(int td = 0; td < TILE_D && (od + td) < out_depth; td++) {
        for(int th = 0; th < TILE_H && (oh + th) < out_height; th++) {
            for(int tw = 0; tw < TILE_W && (ow + tw) < out_width; tw++) {
                float final_val = sum[td][th][tw];
                if(bias != nullptr) {
                    final_val += bias[oc];
                }
                int out_idx = ((batch_id * out_channels + oc) * out_depth + (od + td)) *
                              out_height * out_width + (oh + th) * out_width + (ow + tw);
                output[out_idx] = final_val;
            }
        }
    }
}