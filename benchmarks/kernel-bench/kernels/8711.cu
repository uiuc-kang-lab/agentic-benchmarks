#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

constexpr int WARP_SIZE = 32;
constexpr int TILE_D = 2, TILE_HW = 4;

__inline__ __device__ float warp_reduce(float val) {
    for(int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void conv_transpose3d_warp_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups, int in_per_group) {

    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = elem_idx / WARP_SIZE;
    int lane_id = elem_idx % WARP_SIZE;

    int n = warp_id / (C_out * outD * outH / TILE_HW * outW / TILE_HW);
    int residual = warp_id % (C_out * outD * outH / TILE_HW * outW / TILE_HW);

    int oc = residual / (outD * outH / TILE_HW * outW / TILE_HW);
    residual %= outD * outH / TILE_HW * outW / TILE_HW;

    int od_base = (residual / (outH / TILE_HW * outW / TILE_HW)) * TILE_D;
    int ohw = residual % (outH / TILE_HW * outW / TILE_HW);
    int oh = (ohw / (outW / TILE_HW)) * TILE_HW;
    int ow = (ohw % (outW / TILE_HW)) * TILE_HW;

    int group = oc / (C_out / groups);
    int in_start = group * in_per_group;
    float sum[TILE_D][TILE_HW][TILE_HW] = {0};

    for(int icg = 0; icg < in_per_group; icg++) {
        int c_in = in_start + icg;
        for(int kd = 0; kd < kernel_d; kd++) {
            for(int kh = 0; kh < kernel_h; kh++) {
                for(int kw = 0; kw < kernel_w; kw++) {
                    float w_val = weight[((((c_in * C_out/groups + oc%in_per_group) * kernel_d) + kd) * kernel_h + kh) * kernel_w + kw];
                    
                    for(int td = 0; td < TILE_D; td++) {
                        int od = od_base + td;
                        if(od >= outD) continue;
                        int d_in = (od - kd + pad_d) / stride_d;
                        if((od - kd + pad_d) % stride_d || d_in < 0 || d_in >= D_in) continue;
                        
                        for(int th = 0; th < TILE_HW; th++) {
                            int h = oh + th;
                            if(h >= outH) continue;
                            int h_in = (h - kh + pad_h) / stride_h;
                            if((h - kh + pad_h) % stride_h || h_in < 0 || h_in >= H_in) continue;
                            
                            for(int tw = 0; tw < TILE_HW; tw++) {
                                int w = ow + tw;
                                if(w >= outW) continue;
                                int w_in = (w - kw + pad_w) / stride_w;
                                if((w - kw + pad_w) % stride_w || w_in < 0 || w_in >= W_in) continue;
                                
                                float in_val = input[((((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in)];
                                sum[td][th][tw] += in_val * w_val;
                            }
                        }
                    }
                }
            }
        }
    }

    __syncthreads();
    
    for(int td = 0; td < TILE_D; td++) {
        for(int th = 0; th < TILE_HW; th++) {
            for(int tw = 0; tw < TILE_HW; tw++) {
                float reduced = warp_reduce(sum[td][th][tw]);
                if(lane_id == 0) {
                    int od = od_base + td;
                    int oh_val = oh + th;
                    int ow_val = ow + tw;
                    if(od < outD && oh_val < outH && ow_val < outW) {
                        int out_idx = ((((n * C_out + oc) * outD + od) * outH + oh_val) * outW) + ow_val;
                        atomicAdd(&output[out_idx], reduced);
                    }
                }
            }
        }
    }
}

// [Rest of the code including bias kernel and pybind remains same as previous optimized version]
