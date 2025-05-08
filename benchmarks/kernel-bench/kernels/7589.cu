#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel for ConvTranspose3d that fuses both grouped and non-grouped paths
__global__ void conv_transpose3d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // may be nullptr
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int iD, int iH, int iW,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int outD, int outH, int outW,
    int groups) {

    int total_elements = batch * out_channels * outD * outH * outW;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    // Decode the linear index to 5D indices: b, oc, d, h, w
    int w = index % outW;
    int tmp = index / outW;
    int h = tmp % outH;
    tmp /= outH;
    int d = tmp % outD;
    tmp /= outD;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;

    // Branch once based on group mode
    if (groups == 1) {
        // Non-grouped convolution: iterate over all input channels
        for (int ic = 0; ic < in_channels; ++ic) {
            // Loop over kernel depth
            for (int kd = 0; kd < kD; ++kd) {
                int id = d + pad_d - kd;
                if (id < 0 || (id % stride_d) != 0) continue;
                id /= stride_d;
                if (id >= iD) continue;
                // Loop over kernel height
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = h + pad_h - kh;
                    if (ih < 0 || (ih % stride_h) != 0) continue;
                    ih /= stride_h;
                    if (ih >= iH) continue;
                    // Loop over kernel width
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = w + pad_w - kw;
                        if (iw < 0 || (iw % stride_w) != 0) continue;
                        iw /= stride_w;
                        if (iw >= iW) continue;
                        // Compute input and weight indices
                        int input_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
                        int weight_idx = (((ic * out_channels + oc) * kD + kd) * kH + kh) * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    } else {
        // Grouped convolution: determine channel grouping
        int out_ch_per_group = out_channels / groups;
        int in_ch_per_group = in_channels / groups;
        int group = oc / out_ch_per_group;
        // Loop only over the input channels in the group
        for (int ic = group * in_ch_per_group; ic < (group + 1) * in_ch_per_group; ++ic) {
            for (int kd = 0; kd < kD; ++kd) {
                int id = d + pad_d - kd;
                if (id < 0 || (id % stride_d) != 0) continue;
                id /= stride_d;
                if (id >= iD) continue;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = h + pad_h - kh;
                    if (ih < 0 || (ih % stride_h) != 0) continue;
                    ih /= stride_h;
                    if (ih >= iH) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = w + pad_w - kw;
                        if (iw < 0 || (iw % stride_w) != 0) continue;
                        iw /= stride_w;
                        if (iw >= iW) continue;
                        int input_idx = (((b * in_channels + ic) * iD + id) * iH + ih) * iW + iw;
                        int weight_ic = ic - group * in_ch_per_group;
                        int oc_local = oc % out_ch_per_group;
                        int weight_idx = (((weight_ic * out_ch_per_group + oc_local) * kD + kd) * kH + kh) * kW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_idx = (((b * out_channels + oc) * outD + d) * outH + h) * outW + w;
    output[out_idx] = sum;
}

// Host forward function
torch::Tensor forward_optimized(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Get input dimensions: [batch, in_channels, iD, iH, iW]
    int batch = x.size(0);
    int in_channels = x.size(1);
    int iD = x.size(2);
    int iH = x.size(3);
    int iW = x.size(4);

    // Get kernel dimensions
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    int opad_d = output_padding[0];
    int opad_h = output_padding[1];
    int opad_w = output_padding[2];

    // Compute output dimensions for conv_transpose3d
    int outD = (iD - 1) * stride_d - 2 * pad_d + kD + opad_d;
    int outH = (iH - 1) * stride_h - 2 * pad_h + kH + opad_h;
    int outW = (iW - 1) * stride_w - 2 * pad_w + kW + opad_w;

    int out_channels = (groups == 1) ? weight.size(1) : weight.size(1) * groups;
    
    auto options = x.options();
    auto output = torch::zeros({batch, out_channels, outD, outH, outW}, options);

    int total_threads = batch * out_channels * outD * outH * outW;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;  // Limit excessive blocks

    conv_transpose3d_optimized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        iD, iH, iW,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        outD, outH, outW,
        groups
    );
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &forward_optimized, "Optimized ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
