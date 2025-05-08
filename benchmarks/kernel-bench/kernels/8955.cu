#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>

#define UNROLL_FACTOR 4
#define WARP_SIZE 32

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * out_channels * out_h * out_w;
    
    // Vectorized initialization with 4-element writes
    for(int i = tid; i < total/4; i += blockDim.x * gridDim.x) {
        const int idx = i * 4;
        const int oc = (idx / (out_h * out_w)) % out_channels;
        const float4 val = {bias[oc], bias[oc], bias[oc], bias[oc]};
        reinterpret_cast<float4*>(output)[i] = val;
    }
    
    // Handle remaining elements
    const int remainder_start = (total/4)*4;
    for(int i = tid + remainder_start; i < total; i += blockDim.x * gridDim.x) {
        const int oc = (i / (out_h * out_w)) % out_channels;
        output[i] = bias[oc];
    }
}

__global__ void conv_transposed2d_optimized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels_per_group,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int out_h,
    const int out_w,
    const int in_channels_per_group) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * in_channels * in_h * in_w;
    if (index >= total) return;

    // Input tensor decomposition
    const int iw = index % in_w;
    const int ih = (index / in_w) % in_h;
    const int c = (index / (in_w * in_h)) % in_channels;
    const int n = index / (in_w * in_h * in_channels);

    const float x_val = __ldg(&x[index]);
    const int group = c / in_channels_per_group;
    const int weight_base = c * out_channels_per_group * kernel_h * kernel_w;
    const int output_batch_offset = n * groups * out_channels_per_group * out_h * out_w;

    // Precompute valid kernel ranges to minimize conditionals
    const int kh_start = max(0, (pad_h - ih * stride_h + dilation_h - 1) / dilation_h);
    const int kh_end = min(kernel_h, (out_h + pad_h - ih * stride_h) / dilation_h + 1);

    const int kw_start = max(0, (pad_w - iw * stride_w + dilation_w - 1) / dilation_w);
    const int kw_end = min(kernel_w, (out_w + pad_w - iw * stride_w) / dilation_w + 1);

    for (int kh = kh_start; kh < kh_end; ++kh) {
        const int out_row = ih * stride_h - pad_h + kh * dilation_h;
        
        #pragma unroll UNROLL_FACTOR
        for (int kw = kw_start; kw < kw_end; ++kw) {
            const int out_col = iw * stride_w - pad_w + kw * dilation_w;
            
            const int kernel_offset = kh * kernel_w + kw;
            const int output_pos_base = output_batch_offset + group * out_channels_per_group * out_h * out_w 
                                      + out_row * out_w + out_col;

            // Process 4 output channels at a time
            for(int oc_base = 0; oc_base < out_channels_per_group; oc_base += UNROLL_FACTOR) {
                float4 weights;
                weights.x = __ldg(&weight[weight_base + (oc_base + 0) * kernel_h * kernel_w + kernel_offset]);
                weights.y = __ldg(&weight[weight_base + (oc_base + 1) * kernel_h * kernel_w + kernel_offset]);
                weights.z = __ldg(&weight[weight_base + (oc_base + 2) * kernel_h * kernel_w + kernel_offset]);
                weights.w = __ldg(&weight[weight_base + (oc_base + 3) * kernel_h * kernel_w + kernel_offset]);

                const int out_offset = oc_base * out_h * out_w;
                atomicAdd(&output[output_pos_base + out_offset], x_val * weights.x);
                atomicAdd(&output[output_pos_base + out_offset + out_h * out_w], x_val * weights.y);
                atomicAdd(&output[output_pos_base + out_offset + 2 * out_h * out_w], x_val * weights.z);
                atomicAdd(&output[output_pos_base + out_offset + 3 * out_h * out_w], x_val * weights.w);
            }
        }
    }
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
    
    if (!bias.has_value() || !bias.value().defined()) {
        bias = at::zeros({weight.size(1) * groups}, weight.options());
    } else {
        bias = bias.value().contiguous();
    }

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    const int out_channels_per_group = weight.size(1);
    const int out_channels = out_channels_per_group * groups;
    const int in_channels_per_group = in_channels / groups;
    
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int pad_h = padding[0];
    const int pad_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];

    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + 1;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + 1;

    auto output = at::empty({batch, out_channels, out_h, out_w}, x.options());

    // Initialize output with bias
    const int total_output = batch * out_channels * out_h * out_w;
    const int threads_init = 256;
    const int blocks_init = (total_output + threads_init * 4 - 1) / (threads_init * 4);
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w);

    // Launch optimized kernel
    const int total_input = batch * in_channels * in_h * in_w;
    const int threads = 256;
    const int blocks = (total_input + threads - 1) / threads;
    conv_transposed2d_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        out_h,
        out_w,
        in_channels_per_group);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2D Transposed Convolution (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}