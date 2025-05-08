#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel that uses a grid-stride loop for processing output elements
__global__ void conv_transposed_3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // may be nullptr
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int kD, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int opad_d, int opad_h, int opad_w,
    int groups, int in_channels_per_group, int out_channels_per_group
) {
    int total = N * C_out * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    // Grid-stride loop to cover all output elements
    for (int out_index = idx; out_index < total; out_index += gridStride) {
        // Decompose out_index into [n, c, d, h, w]
        int w = out_index % W_out;
        int tmp = out_index / W_out;
        int h = tmp % H_out;
        tmp /= H_out;
        int d = tmp % D_out;
        tmp /= D_out;
        int c = tmp % C_out;
        tmp /= C_out;
        int n = tmp;  // remaining index is batch index

        float sum = 0.0f;

        if (groups == 1) {
            // Non-grouped convolution: weight shape assumed to be [C_in, C_out, kD, kH, kW]
            for (int ic = 0; ic < C_in; ++ic) {
                for (int kd = 0; kd < kD; ++kd) {
                    int id_temp = d + pad_d - kd;
                    if (id_temp % stride_d != 0) continue;
                    int id = id_temp / stride_d;
                    if (id < 0 || id >= D_in) continue;
                    for (int kh = 0; kh < kH; ++kh) {
                        int ih_temp = h + pad_h - kh;
                        if (ih_temp % stride_h != 0) continue;
                        int ih = ih_temp / stride_h;
                        if (ih < 0 || ih >= H_in) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            int iw_temp = w + pad_w - kw;
                            if (iw_temp % stride_w != 0) continue;
                            int iw = iw_temp / stride_w;
                            if (iw < 0 || iw >= W_in) continue;
                            int input_idx = (((n * C_in + ic) * D_in + id) * H_in + ih) * W_in + iw;
                            int weight_idx = ((((ic * C_out) + c) * kD + kd) * kH + kh) * kW + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        } else {
            // Grouped convolution: weight shape is [in_channels_per_group, out_channels_per_group, kD, kH, kW]
            int group = c / out_channels_per_group;
            for (int ic = 0; ic < in_channels_per_group; ++ic) {
                int ic_global = group * in_channels_per_group + ic;
                for (int kd = 0; kd < kD; ++kd) {
                    int id_temp = d + pad_d - kd;
                    if (id_temp % stride_d != 0) continue;
                    int id = id_temp / stride_d;
                    if (id < 0 || id >= D_in) continue;
                    for (int kh = 0; kh < kH; ++kh) {
                        int ih_temp = h + pad_h - kh;
                        if (ih_temp % stride_h != 0) continue;
                        int ih = ih_temp / stride_h;
                        if (ih < 0 || ih >= H_in) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            int iw_temp = w + pad_w - kw;
                            if (iw_temp % stride_w != 0) continue;
                            int iw = iw_temp / stride_w;
                            if (iw < 0 || iw >= W_in) continue;
                            int input_idx = (((n * C_in + ic_global) * D_in + id) * H_in + ih) * W_in + iw;
                            int oc_local = c % out_channels_per_group;
                            int weight_idx = ((((ic * out_channels_per_group) + oc_local) * kD + kd) * kH + kh) * kW + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c];
        }
        int output_idx = (((n * C_out + c) * D_out + d) * H_out + h) * W_out + w;
        output[output_idx] = sum;
    }
}

// Host forward function
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Input dimensions: [N, C_in, D_in, H_in, W_in]
    int N = x.size(0);
    int C_in = x.size(1);
    int D_in = x.size(2);
    int H_in = x.size(3);
    int W_in = x.size(4);

    // Weight dimensions
    // For non-grouped: weight shape is [C_in, C_out, kD, kH, kW]
    // For grouped convolution: weight shape is [in_channels_per_group, out_channels_per_group, kD, kH, kW]
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

    // Compute output dimensions for transposed convolution
    int D_out = (D_in - 1) * stride_d - 2 * pad_d + kD + opad_d;
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + opad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + opad_w;

    int C_out;
    int in_channels_per_group;
    int out_channels_per_group;
    if (groups == 1) {
        C_out = weight.size(1); // non-grouped convolution
        in_channels_per_group = C_in;
        out_channels_per_group = C_out;
    } else {
        in_channels_per_group = C_in / groups;
        out_channels_per_group = weight.size(1);
        C_out = groups * out_channels_per_group;
    }

    auto options = x.options();
    auto output_tensor = torch::zeros({N, C_out, D_out, H_out, W_out}, options);

    int total_threads = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    conv_transposed_3d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output_tensor.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        opad_d, opad_h, opad_w,
        groups, in_channels_per_group, out_channels_per_group
    );
    cudaDeviceSynchronize();
    return output_tensor;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function using stride loops",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
