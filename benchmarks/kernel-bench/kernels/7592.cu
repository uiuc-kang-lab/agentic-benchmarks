#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Threshold for using optimized kernel vs ATen implementation
#define CHANNEL_THRESHOLD 32
#define KERNEL_SIZE_THRESHOLD 7

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_channels, int out_channels,
    int iD, int iH, int iW,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups) {

    int total_elements = batch * out_channels * outD * outH * outW;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    int w = index % outW;
    int tmp = index / outW;
    int h = tmp % outH;
    tmp = tmp / outH;
    int d = tmp % outD;
    tmp = tmp / outD;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;

    if (groups == 1) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kD; ++kd) {
                int id = (d + pad_d - kd * stride_d);
                if (id < 0 || id >= iD || id % stride_d != 0) continue;
                id /= stride_d;

                for (int kh = 0; kh < kH; ++kh) {
                    int ih = (h + pad_h - kh * stride_h);
                    if (ih < 0 || ih >= iH || ih % stride_h != 0) continue;
                    ih /= stride_h;

                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = (w + pad_w - kw * stride_w);
                        if (iw < 0 || iw >= iW || iw % stride_w != 0) continue;
                        iw /= stride_w;

                        int in_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
                        int w_idx = ((((oc) * in_channels + ic) * kD + kd) * kH + kh) * kW + kw;
                        
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    } else {
        int out_channels_per_group = out_channels / groups;
        int in_channels_per_group = in_channels / groups;
        int group = oc / out_channels_per_group;
        int oc_within_group = oc % out_channels_per_group;
        
        for (int ic = 0; ic < in_channels_per_group; ++ic) {
            int ic_global = group * in_channels_per_group + ic;
            
            for (int kd = 0; kd < kD; ++kd) {
                int id = (d + pad_d - kd * stride_d);
                if (id < 0 || id >= iD || id % stride_d != 0) continue;
                id /= stride_d;

                for (int kh = 0; kh < kH; ++kh) {
                    int ih = (h + pad_h - kh * stride_h);
                    if (ih < 0 || ih >= iH || ih % stride_h != 0) continue;
                    ih /= stride_h;

                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = (w + pad_w - kw * stride_w);
                        if (iw < 0 || iw >= iW || iw % stride_w != 0) continue;
                        iw /= stride_w;

                        int in_idx = (((b * in_channels + ic_global) * iD + id) * iH + ih) * iW + iw;
                        int w_idx = ((((oc_within_group) * in_channels_per_group + ic) * kD + kd) * kH + kh) * kW + kw;
                        
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }
    output[(((b * out_channels + oc) * outD + d) * outH + h) * outW + w] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {
    
    if (x.size(1) < CHANNEL_THRESHOLD || 
        weight.size(2) > KERNEL_SIZE_THRESHOLD ||
        weight.size(3) > KERNEL_SIZE_THRESHOLD ||
        weight.size(4) > KERNEL_SIZE_THRESHOLD) {
        return at::conv_transpose3d(
            x, weight,
            bias ? *bias : torch::Tensor(),
            stride, padding, output_padding,
            groups, {1,1,1}
        );
    }

    int batch = x.size(0);
    int in_channels = x.size(1);
    int out_channels = weight.size(1) * (groups == 1 ? 1 : groups);
    
    int outD = (x.size(2) - 1) * stride[0] - 2 * padding[0] + weight.size(2) + output_padding[0];
    int outH = (x.size(3) - 1) * stride[1] - 2 * padding[1] + weight.size(3) + output_padding[1];
    int outW = (x.size(4) - 1) * stride[2] - 2 * padding[2] + weight.size(4) + output_padding[2];

    auto output = torch::zeros({batch, out_channels, outD, outH, outW}, x.options());

    const int threads = 256;
    const int blocks = (batch * out_channels * outD * outH * outW + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float) * 2;

    conv_transpose3d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        x.size(2), x.size(3), x.size(4),
        weight.size(2), weight.size(3), weight.size(4),
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        outD, outH, outW,
        groups
    );

    return output;
}