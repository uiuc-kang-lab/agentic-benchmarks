#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/util/Optional.h>
#include <stdio.h>

__global__ void initialize_output_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int batch,
    const int out_channels,
    const int out_h,
    const int out_w) {
    
    // Use 4-wide vectorized loads/stores where possible
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    
    // Handle main aligned portion with float4
    int aligned_total = total / 4 * 4;
    for (int i = tid * 4; i < aligned_total; i += blockDim.x * gridDim.x * 4) {
        if (i + 3 >= total) break;
        
        // Compute channel indices for each element in the float4
        int idx0 = i;
        int idx1 = i + 1;
        int idx2 = i + 2;
        int idx3 = i + 3;
        
        int oc0 = (idx0 / out_w / out_h) % out_channels;
        int oc1 = (idx1 / out_w / out_h) % out_channels;
        int oc2 = (idx2 / out_w / out_h) % out_channels;
        int oc3 = (idx3 / out_w / out_h) % out_channels;
        
        float4 bias_val;
        bias_val.x = __ldg(&bias[oc0]);
        bias_val.y = __ldg(&bias[oc1]);
        bias_val.z = __ldg(&bias[oc2]);
        bias_val.w = __ldg(&bias[oc3]);
        
        reinterpret_cast<float4*>(output)[i/4] = bias_val;
    }
    
    // Handle remaining elements
    for (int i = tid + aligned_total; i < total; i += blockDim.x * gridDim.x) {
        int oc = (i / out_w / out_h) % out_channels;
        output[i] = __ldg(&bias[oc]);
    }
}

__global__ void conv_transposed2d_scatter_atomic_kernel(
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

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * in_channels * in_h * in_w;
    if (index >= total) return;

    // Decode input index
    int iw = index % in_w;
    int tmp = index / in_w;
    int ih = tmp % in_h;
    tmp = tmp / in_h;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    float x_val = __ldg(&x[index]);
    int group = c / in_channels_per_group;
    
    // Pre-compute base indices for weight access
    int weight_base = c * (out_channels_per_group * kernel_h * kernel_w);
    int output_batch_offset = n * (groups * out_channels_per_group * out_h * out_w);
    int group_offset = group * out_channels_per_group;

    #pragma unroll 4
    for (int kh = 0; kh < kernel_h; kh++) {
        int out_row = ih * stride_h - pad_h + kh * dilation_h;
        if (out_row < 0 || out_row >= out_h) continue;

        #pragma unroll 4
        for (int kw = 0; kw < kernel_w; kw++) {
            int out_col = iw * stride_w - pad_w + kw * dilation_w;
            if (out_col < 0 || out_col >= out_w) continue;

            int kernel_offset = kh * kernel_w + kw;
            
            // Process output channels in chunks of 4 where possible
            int aligned_channels = (out_channels_per_group / 4) * 4;
            
            // Handle aligned channels
            for (int oc_base = 0; oc_base < aligned_channels; oc_base += 4) {
                float4 weight_vec;
                weight_vec.x = __ldg(&weight[weight_base + (oc_base + 0) * (kernel_h * kernel_w) + kernel_offset]);
                weight_vec.y = __ldg(&weight[weight_base + (oc_base + 1) * (kernel_h * kernel_w) + kernel_offset]);
                weight_vec.z = __ldg(&weight[weight_base + (oc_base + 2) * (kernel_h * kernel_w) + kernel_offset]);
                weight_vec.w = __ldg(&weight[weight_base + (oc_base + 3) * (kernel_h * kernel_w) + kernel_offset]);

                int out_base = output_batch_offset + (group_offset + oc_base) * (out_h * out_w) + out_row * out_w + out_col;
                
                atomicAdd(&output[out_base], x_val * weight_vec.x);
                atomicAdd(&output[out_base + out_h * out_w], x_val * weight_vec.y);
                atomicAdd(&output[out_base + 2 * out_h * out_w], x_val * weight_vec.z);
                atomicAdd(&output[out_base + 3 * out_h * out_w], x_val * weight_vec.w);
            }
            
            // Handle remaining channels
            for (int oc_offset = aligned_channels; oc_offset < out_channels_per_group; oc_offset++) {
                float weight_val = __ldg(&weight[weight_base + oc_offset * (kernel_h * kernel_w) + kernel_offset]);
                int out_index = output_batch_offset + (group_offset + oc_offset) * (out_h * out_w) + 
                               out_row * out_w + out_col;
                atomicAdd(&output[out_index], x_val * weight_val);
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

    int total_output = batch * out_channels * out_h * out_w;
    const int threads_init = 256;
    const int blocks_init = (total_output + threads_init * 4 - 1) / (threads_init * 4);
    
    initialize_output_kernel<<<blocks_init, threads_init>>>(
        output.data_ptr<float>(),
        bias.value().data_ptr<float>(),
        batch,
        out_channels,
        out_h,
        out_w);

    int total_input = batch * in_channels * in_h * in_w;
    const int threads_scatter = 256;
    const int blocks_scatter = (total_input + threads_scatter - 1) / threads_scatter;
    
    conv_transposed2d_scatter_atomic_kernel<<<blocks_scatter, threads_scatter>>>(
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
    m.def("forward", &forward, "2D Transposed Convolution using Scatter with Atomics (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}