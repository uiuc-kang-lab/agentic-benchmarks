#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = 4;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ int gcd(int a, int b) {
  while(b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

__device__ __forceinline__ int my_min(int a, int b) {
  return a < b ? a : b;
}

struct SharedMemory {
    __device__ inline float* getPointer() {
        extern __shared__ float s[];
        return s;
    }
};

__global__ void conv_transpose2d_kernel_optimized(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group) {

    SharedMemory smem;
    float* shared_input = smem.getPointer();
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = thread_id * ELEMENTS_PER_THREAD;
    int total = batch * out_channels * out_h * out_w;

    float out_vals[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = base_idx + i;
        if (idx < total) {
            int oc = (idx / (out_h * out_w)) % out_channels;
            out_vals[i] = __ldg(&bias[oc]);
        }
    }

    #pragma unroll
    for (int elem = 0; elem < ELEMENTS_PER_THREAD; elem++) {
        int idx = base_idx + elem;
        if (idx >= total) continue;

        int ow = idx % out_w;
        int tmp = idx / out_w;
        int oh = tmp % out_h;
        tmp = tmp / out_h;
        int oc = tmp % out_channels;
        int n = tmp / out_channels;
        
        int g = oc / out_channels_per_group;
        
        int candidate_h = oh + pad_h;
        int candidate_w = ow + pad_w;
        
        int mod_h = candidate_h % stride_h;
        int mod_w = candidate_w % stride_w;
        
        int offset_kh = -1, offset_kw = -1;
        #pragma unroll
        for (int k = 0; k < stride_h; k++) {
            if ((k * dilation_h) % stride_h == mod_h) {
                offset_kh = k;
                break;
            }
        }
        
        #pragma unroll
        for (int k = 0; k < stride_w; k++) {
            if ((k * dilation_w) % stride_w == mod_w) {
                offset_kw = k;
                break;
            }
        }

        int step_kh = stride_h / gcd(stride_h, dilation_h);
        int step_kw = stride_w / gcd(stride_w, dilation_w);
        
        int kh_bound = candidate_h / dilation_h + 1;
        int kw_bound = candidate_w / dilation_w + 1;
        int kh_end = my_min(kernel_h, kh_bound);
        int kw_end = my_min(kernel_w, kw_bound);

        #pragma unroll 4
        for (int kh = offset_kh; kh >= 0 && kh < kh_end; kh += step_kh) {
            int h_in = (candidate_h - kh * dilation_h) / stride_h;
            if (h_in < 0 || h_in >= in_h) continue;

            #pragma unroll 4
            for (int kw = offset_kw; kw >= 0 && kw < kw_end; kw += step_kw) {
                int w_in = (candidate_w - kw * dilation_w) / stride_w;
                if (w_in < 0 || w_in >= in_w) continue;

                #pragma unroll
                for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c += 4) {
                    float4 x_vec, w_vec;
                    
                    int x_base = ((n * in_channels + c) * in_h + h_in) * in_w + w_in;
                    int w_base = ((c * out_channels_per_group + (oc - g * out_channels_per_group)) * kernel_h + kh) * kernel_w + kw;
                    
                    x_vec = *reinterpret_cast<const float4*>(&x[x_base]);
                    w_vec = *reinterpret_cast<const float4*>(&weight[w_base]);
                    
                    out_vals[elem] += x_vec.x * w_vec.x + x_vec.y * w_vec.y + 
                                    x_vec.z * w_vec.z + x_vec.w * w_vec.w;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = base_idx + i;
        if (idx < total) {
            int ow = idx % out_w;
            int tmp = idx / out_w;
            int oh = tmp % out_h;
            tmp = tmp / out_h;
            int oc = tmp % out_channels;
            int n = tmp / out_channels;
            
            int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
            output[out_idx] = out_vals[i];
        }
    }
}