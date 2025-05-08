#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 128
#define BLOCK_SIZE 256

__global__ void conv_transposed1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_width,
    const int groups) {
    
    // Calculate output position
    const int tidx = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tidx;
    
    if (idx >= batch_size * out_channels * output_width) return;
    
    const int out_pos = idx % output_width;
    const int out_ch = (idx / output_width) % out_channels;
    const int batch = idx / (output_width * out_channels);
    
    // Calculate group information
    const int group_size = in_channels / groups;
    const int group = out_ch / (out_channels / groups);
    const int in_ch_start = group * group_size;
    const int in_ch_end = in_ch_start + group_size;
    
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int k = 0; k < kernel_size; k++) {
        const int in_pos = (out_pos + padding - k) / stride;
        if (in_pos >= 0 && in_pos < input_width && ((out_pos + padding - k) % stride == 0)) {
            for (int ic = in_ch_start; ic < in_ch_end; ic += 4) {
                if (ic + 3 < in_ch_end) {
                    float4 in_val;
                    float4 weight_val;
                    
                    in_val.x = __ldg(&input[batch * in_channels * input_width + ic * input_width + in_pos]);
                    in_val.y = __ldg(&input[batch * in_channels * input_width + (ic+1) * input_width + in_pos]);
                    in_val.z = __ldg(&input[batch * in_channels * input_width + (ic+2) * input_width + in_pos]);
                    in_val.w = __ldg(&input[batch * in_channels * input_width + (ic+3) * input_width + in_pos]);
                    
                    weight_val.x = __ldg(&weight[ic * out_channels * kernel_size + out_ch * kernel_size + k]);
                    weight_val.y = __ldg(&weight[(ic+1) * out_channels * kernel_size + out_ch * kernel_size + k]);
                    weight_val.z = __ldg(&weight[(ic+2) * out_channels * kernel_size + out_ch * kernel_size + k]);
                    weight_val.w = __ldg(&weight[(ic+3) * out_channels * kernel_size + out_ch * kernel_size + k]);
                    
                    sum += in_val.x * weight_val.x + in_val.y * weight_val.y + 
                          in_val.z * weight_val.z + in_val.w * weight_val.w;
                } else {
                    for (int i = 0; i < min(4, in_ch_end - ic); i++) {
                        float in_val = __ldg(&input[batch * in_channels * input_width + (ic+i) * input_width + in_pos]);
                        float weight_val = __ldg(&weight[(ic+i) * out_channels * kernel_size + out_ch * kernel_size + k]);
                        sum += in_val * weight_val;
                    }
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += __ldg(&bias[out_ch]);
    }
    
    output[idx] = sum;
}

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_CUDA(x);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);
    
    const auto batch_size = x.size(0);
    const auto in_channels = x.size(1);
    const auto input_width = x.size(2);
    const auto kernel_size = weight.size(2);
    const auto out_channels = weight.size(1) * groups;
    
    const auto output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_width}, x.options());
    
    const int total_elements = batch_size * out_channels * output_width;
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    conv_transposed1d_kernel<<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_width,
        kernel_size,
        stride,
        padding,
        output_width,
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CUDA)");
}