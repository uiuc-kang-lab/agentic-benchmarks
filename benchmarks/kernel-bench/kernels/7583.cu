#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Device function for performing the convolution
__device__ void conv_transpose3d_computation(
    const float* __restrict__ input, const float* __restrict__ weight,
    float& sum, int n, int ic, int oc, int D, int H, int W,
    int id, int ih, int iw, int kd, int kh, int kw,
    int stride_d, int stride_h, int stride_w,
    int iD, int iH, int iW,
    int kD, int kH, int kW) {
    // Iterate through kernel dimensions
    #pragma unroll 8
    for (int kd_unrolled = 0; kd_unrolled < kD; ++kd_unrolled) {
        int id_candidate = id + kd - kd_unrolled * stride_d;
        if (id_candidate < 0 || id_candidate >= iD) continue;
        #pragma unroll 8
        for (int kh_unrolled = 0; kh_unrolled < kH; ++kh_unrolled) {
            int ih_candidate = ih + kh - kh_unrolled * stride_h;
            if (ih_candidate < 0 || ih_candidate >= iH) continue;
            #pragma unroll 8
            for (int kw_unrolled = 0; kw_unrolled < kW; ++kw_unrolled) {
                int iw_candidate = iw + kw - kw_unrolled * stride_w;
                if (iw_candidate < 0 || iw_candidate >= iW) continue;

                // Calculate input and weight index
                int input_idx = (((n * iD + id_candidate) * iH + ih_candidate) * iW + iw_candidate);
                int weight_idx = (((((oc * iD) + kd_unrolled) * kH + kh_unrolled) * kW + kw_unrolled) * iD + ic);
                
                // Accumulate sum
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
}

// CUDA kernel rewritten to use a modular computation function
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
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

    // Decode output index into (b, oc, d, h, w)
    int w = index % outW;
    int tmp = index / outW;
    int h = tmp % outH;
    tmp = tmp / outH;
    int d = tmp % outD;
    tmp = tmp / outD;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float sum = 0.0f;

    // Case for non-grouped convolution
    if (groups == 1) {
        for (int ic = 0; ic < in_channels; ++ic) {
            conv_transpose3d_computation(
                input, weight, sum, b, ic, oc, d, h, w, d, h, w, kD, kH, kW,
                stride_d, stride_h, stride_w, iD, iH, iW, kD, kH, kW);
        }
    } else {
        // Grouped convolution
        int out_channels_per_group = out_channels / groups;
        int in_channels_per_group = in_channels / groups;
        int group = oc / out_channels_per_group;
        for (int ic = group * in_channels_per_group; ic < (group + 1) * in_channels_per_group; ++ic) {
            conv_transpose3d_computation(
                input, weight, sum, b, ic, oc, d, h, w, d, h, w, kD, kH, kW,
                stride_d, stride_h, stride_w, iD, iH, iW, kD, kH, kW);
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }
    // Write the result into output tensor
    output[(((b * out_channels + oc) * outD + d) * outH + h) * outW + w] = sum;
}

// Host forward function that prepares parameters and launches the kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Input dimensions
    int batch = x.size(0);
    int in_channels = x.size(1);
    int iD = x.size(2);
    int iH = x.size(3);
    int iW = x.size(4);

    // Weight dimensions
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int stride_d = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];

    int pad_d = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];

    int out_pad_d = output_padding[0];
    int out_pad_h = output_padding[1];
    int out_pad_w = output_padding[2];

    // Compute output dimensions
    int outD = (iD - 1) * stride_d - 2 * pad_d + kD + out_pad_d;
    int outH = (iH - 1) * stride_h - 2 * pad_h + kH + out_pad_h;
    int outW = (iW - 1) * stride_w - 2 * pad_w + kW + out_pad_w;

    int out_channels = (groups == 1) ? weight.size(1) : weight.size(1) * groups;

    auto options = x.options();
    auto output = torch::zeros({batch, out_channels, outD, outH, outW}, options);

    int total_threads = batch * out_channels * outD * outH * outW;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* out_ptr = output.data_ptr<float>();

    conv_transpose3d_kernel<<<blocks, threads>>>(x_ptr, w_ptr, b_ptr, out_ptr,
        batch, in_channels, out_channels,
        iD, iH, iW,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        outD, outH, outW,
        groups);

    cudaDeviceSynchronize();
    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function with modular device functions",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}