#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define USE_SHARED_MEM_THRESHOLD 512
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<bool UseSharedMem>
__global__ void optimized_conv3d_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_d, const int in_h, const int in_w,
    const int out_d, const int out_h, const int out_w,
    const int k_d, const int k_h, const int k_w,
    const int s_d, const int s_h, const int s_w,
    const int p_d, const int p_h, const int p_w,
    const int groups,
    const int channels_per_group_in,
    const int channels_per_group_out) {
    
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int total = batch * out_channels * out_d * out_h * out_w;
    
    for (int idx = blockIdx.x * blockDim.x + tid; idx < total; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        const int w_out = tmp % out_w; tmp /= out_w;
        const int h_out = tmp % out_h; tmp /= out_h;
        const int d_out = tmp % out_d; tmp /= out_d;
        const int oc = tmp % out_channels; tmp /= out_channels;
        const int n = tmp;

        float sum = (bias != nullptr) ? bias[oc] : 0.0f;
        
        const int group = oc / channels_per_group_out;
        const int oc_in_group = oc % channels_per_group_out;
        
        if (UseSharedMem) {
            float* shared_weights = shared_mem;
            #pragma unroll
            for (int kd = 0; kd < k_d; kd++) {
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        if (tid < channels_per_group_in) {
                            int widx = tid * (k_d * k_h * k_w) + (kd * k_h * k_w + kh * k_w + kw);
                            shared_weights[widx] = weight[group * channels_per_group_in * channels_per_group_out * k_d * k_h * k_w + 
                                                       oc_in_group * k_d * k_h * k_w + widx];
                        }
                    }
                }
            }
            __syncthreads();
            
            const int d_base = d_out + p_d;
            const int h_base = h_out + p_h;
            const int w_base = w_out + p_w;
            
            #pragma unroll 4
            for (int ic = 0; ic < channels_per_group_in; ic++) {
                const int in_channel = group * channels_per_group_in + ic;
                
                for (int kd = 0; kd < k_d; kd++) {
                    const int in_d_idx = (d_base - kd) / s_d;
                    if ((d_base - kd) % s_d == 0 && in_d_idx >= 0 && in_d_idx < in_d) {
                        
                        for (int kh = 0; kh < k_h; kh++) {
                            const int in_h_idx = (h_base - kh) / s_h;
                            if ((h_base - kh) % s_h == 0 && in_h_idx >= 0 && in_h_idx < in_h) {
                                
                                for (int kw = 0; kw < k_w; kw++) {
                                    const int in_w_idx = (w_base - kw) / s_w;
                                    if ((w_base - kw) % s_w == 0 && in_w_idx >= 0 && in_w_idx < in_w) {
                                        
                                        const float in_val = input[n * in_channels * in_d * in_h * in_w +
                                                                 in_channel * in_d * in_h * in_w +
                                                                 in_d_idx * in_h * in_w +
                                                                 in_h_idx * in_w + in_w_idx];
                                                                 
                                        const int weight_offset = ic * k_d * k_h * k_w +
                                                                kd * k_h * k_w + kh * k_w + kw;
                                        
                                        sum += in_val * shared_weights[weight_offset];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            const int d_base = d_out + p_d;
            const int h_base = h_out + p_h;
            const int w_base = w_out + p_w;
            
            #pragma unroll 4
            for (int ic = 0; ic < channels_per_group_in; ic++) {
                const int in_channel = group * channels_per_group_in + ic;
                
                for (int kd = 0; kd < k_d; kd++) {
                    const int in_d_idx = (d_base - kd) / s_d;
                    if ((d_base - kd) % s_d == 0 && in_d_idx >= 0 && in_d_idx < in_d) {
                        
                        for (int kh = 0; kh < k_h; kh++) {
                            const int in_h_idx = (h_base - kh) / s_h;
                            if ((h_base - kh) % s_h == 0 && in_h_idx >= 0 && in_h_idx < in_h) {
                                
                                for (int kw = 0; kw < k_w; kw++) {
                                    const int in_w_idx = (w_base - kw) / s_w;
                                    if ((w_base - kw) % s_w == 0 && in_w_idx >= 0 && in_w_idx < in_w) {
                                        
                                        const float in_val = input[n * in_channels * in_d * in_h * in_w +
                                                                 in_channel * in_d * in_h * in_w +
                                                                 in_d_idx * in_h * in_w +
                                                                 in_h_idx * in_w + in_w_idx];
                                                                 
                                        const int weight_idx = in_channel * channels_per_group_out * k_d * k_h * k_w +
                                                             oc_in_group * k_d * k_h * k_w +
                                                             kd * k_h * k_w + kh * k_w + kw;
                                                             
                                        sum += in_val * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}