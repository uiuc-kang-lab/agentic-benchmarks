#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

#define TILE_OC 32
#define TILE_SP 8

// Device function to compute GCD
__device__ __forceinline__ int gcd_device(int a, int b) {
    while(b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Device function to compute minimum
__device__ __forceinline__ int min_device(int a, int b) {
    return (a < b) ? a : b;
}

// Device function to compute kernel offsets and steps
__device__ __forceinline__ void compute_kernel_params(
    int candidate,
    int stride,
    int dilation,
    int kernel_size,
    int& offset,
    int& step,
    int& end) {
    
    offset = -1;
    int mod = candidate % stride;
    
    #pragma unroll
    for (int k = 0; k < stride; k++) {
        if ((k * dilation) % stride == mod) {
            offset = k;
            break;
        }
    }
    
    step = stride / gcd_device(stride, dilation);
    int bound = candidate / dilation + 1;
    end = min_device(kernel_size, bound);
}

// Device function to compute input indices
__device__ __forceinline__ bool compute_input_indices(
    int candidate,
    int k,
    int dilation,
    int stride,
    int in_size,
    int& idx) {
    
    int in_candidate = candidate - k * dilation;
    if (in_candidate < 0 || (in_candidate % stride) != 0) return false;
    
    idx = in_candidate / stride;
    return (idx >= 0 && idx < in_size);
}

// Device function to compute output value for a single position
__device__ __forceinline__ float compute_output_value(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int n, int oc, int oh, int ow,
    int in_channels, int in_h, int in_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int g, int in_channels_per_group, int out_channels_per_group) {
    
    float result = 0.0f;
    
    int candidate_h = oh + pad_h;
    int candidate_w = ow + pad_w;
    
    int offset_kh, step_kh, kh_end;
    int offset_kw, step_kw, kw_end;
    
    compute_kernel_params(candidate_h, stride_h, dilation_h, kernel_h, offset_kh, step_kh, kh_end);
    compute_kernel_params(candidate_w, stride_w, dilation_w, kernel_w, offset_kw, step_kw, kw_end);
    
    #pragma unroll
    for (int kh = offset_kh; kh >= 0 && kh < kh_end; kh += step_kh) {
        int ih;
        if (!compute_input_indices(candidate_h, kh, dilation_h, stride_h, in_h, ih)) continue;
        
        #pragma unroll
        for (int kw = offset_kw; kw >= 0 && kw < kw_end; kw += step_kw) {
            int iw;
            if (!compute_input_indices(candidate_w, kw, dilation_w, stride_w, in_w, iw)) continue;
            
            #pragma unroll
            for (int c = g * in_channels_per_group; c < (g + 1) * in_channels_per_group; c++) {
                int x_idx = ((n * in_channels + c) * in_h + ih) * in_w + iw;
                int w_idx = ((c * out_channels_per_group + (oc - g * out_channels_per_group)) * kernel_h + kh) * kernel_w + kw;
                
                result += __ldg(&x[x_idx]) * __ldg(&weight[w_idx]);
            }
        }
    }
    
    return result;
}

__global__ void conv_transpose2d_kernel_modular(
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
    
    int oc = blockIdx.x * TILE_OC + threadIdx.x;
    int sp_idx = blockIdx.y * TILE_SP + threadIdx.y;
    int n = blockIdx.z;
    
    if (oc >= out_channels || sp_idx >= (out_h * out_w)) return;
    
    int oh = sp_idx / out_w;
    int ow = sp_idx % out_w;
    
    __shared__ float s_bias[TILE_OC];
    if (threadIdx.y == 0 && oc < out_channels) {
        s_bias[threadIdx.x] = __ldg(&bias[oc]);
    }
    __syncthreads();
    
    float out_val = s_bias[threadIdx.x];
    int g = oc / out_channels_per_group;
    
    out_val += compute_output_value(
        x, weight, n, oc, oh, ow,
        in_channels, in_h, in_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        g, in_channels_per_group, out_channels_per_group);
    
    int out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = out_val;
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int groups) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    if (bias.has_value() && bias.value().defined())
        bias = bias.value().contiguous();
    
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;
    
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({out_channels}, weight.options());
    }
    
    auto output = at::zeros({batch, out_channels, out_h, out_w}, x.options());
    
    dim3 block(TILE_OC, TILE_SP);
    dim3 grid((out_channels + TILE_OC - 1) / TILE_OC,
              (out_h * out_w + TILE_SP - 1) / TILE_SP,
              batch);
    
    conv_transpose2d_kernel_modular<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels / groups,
        out_channels_per_group
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular 2D Transposed Convolution (CUDA)");
}