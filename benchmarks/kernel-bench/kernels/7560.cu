#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv3d_uniform_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int N, const int in_channels, const int in_depth, const int in_height, const int in_width,
    const int out_channels, const int out_depth, const int out_height, const int out_width,
    const int kT, const int kH, const int kW,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int groups
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * out_channels * out_depth * out_height * out_width;
    if (idx >= total) return;

    // Decode output position
    const int w_out = idx % out_width;
    const int h_out = (idx / out_width) % out_height;
    const int d_out = (idx / (out_width * out_height)) % out_depth;
    const int c_out = (idx / (out_width * out_height * out_depth)) % out_channels;
    const int n = idx / (out_width * out_height * out_depth * out_channels);

    // Determine group and channels per group
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int group = c_out / out_channels_per_group;
    const int out_c_local = c_out % out_channels_per_group;

    scalar_t sum = 0;

    // Pre-compute base indices and bounds for input coordinates
    const int d_in_base = d_out + pad_d;
    const int h_in_base = h_out + pad_h;
    const int w_in_base = w_out + pad_w;

    // Loop over input channels for current group
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        const int input_channel = group * in_channels_per_group + ic;
        
        // Compute weight base index
        const int weight_base = ((input_channel * out_channels_per_group + out_c_local) * kT) * kH * kW;

        // Loop over kernel dimensions with minimal branching
        for (int kd = 0; kd < kT; kd++) {
            const int d_in_tmp = d_in_base - kd;
            const bool valid_d = (d_in_tmp % stride_d == 0);
            const int d_in = d_in_tmp / stride_d;
            const bool d_in_bounds = (d_in >= 0 && d_in < in_depth);

            if (valid_d && d_in_bounds) {
                for (int kh = 0; kh < kH; kh++) {
                    const int h_in_tmp = h_in_base - kh;
                    const bool valid_h = (h_in_tmp % stride_h == 0);
                    const int h_in = h_in_tmp / stride_h;
                    const bool h_in_bounds = (h_in >= 0 && h_in < in_height);

                    if (valid_h && h_in_bounds) {
                        for (int kw = 0; kw < kW; kw++) {
                            const int w_in_tmp = w_in_base - kw;
                            const bool valid_w = (w_in_tmp % stride_w == 0);
                            const int w_in = w_in_tmp / stride_w;
                            const bool w_in_bounds = (w_in >= 0 && w_in < in_width);

                            if (valid_w && w_in_bounds) {
                                const int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * 
                                                     in_height + h_in) * in_width + w_in;
                                const int weight_idx = weight_base + (kd * kH + kh) * kW + kw;
                                
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[idx] = sum;
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int kT = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int out_channels = weight.size(1) * groups;
    const int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    const int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    const int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    const int total_elements = N * out_channels * out_depth * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_uniform_kernel", ([&] {
        transposed_conv3d_uniform_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with uniform control flow",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}