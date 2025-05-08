#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void transposed_conv3d_optimized_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int in_channels, int in_depth, int in_height, int in_width,
    int out_channels, int out_depth, int out_height, int out_width,
    int kT, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int out_pad_d, int out_pad_h, int out_pad_w,
    int groups
) {
    const int warpSize = 32;
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int total_output_elements = N * out_channels * out_depth * out_height * out_width;
    if (global_thread_id >= total_output_elements) return;

    int out_index = global_thread_id;

    int w_out = out_index % out_width;
    out_index /= out_width;
    int h_out = out_index % out_height;
    out_index /= out_height;
    int d_out = out_index % out_depth;
    out_index /= out_depth;
    int c_out = out_index % out_channels;
    int n = out_index / out_channels;

    int group = c_out / (out_channels / groups);
    int out_c_local = c_out % (out_channels / groups);

    int in_channels_per_group = in_channels / groups;
    int total_iters = in_channels_per_group * (kT * kH * kW);

    scalar_t sum = 0;
    for (int i = threadIdx.x; i < total_iters; i += blockDim.x) {
        int ic = i / (kT * kH * kW);
        int rem = i % (kT * kH * kW);
        int kd = rem / (kH * kW);
        int rem2 = rem % (kH * kW);
        int kh = rem2 / kW;
        int kw = rem2 % kW;

        int input_channel = group * in_channels_per_group + ic;

        int d_in_tmp = d_out + pad_d - kd;
        if (d_in_tmp % stride_d != 0) continue;
        int d_in = d_in_tmp / stride_d;
        if (d_in < 0 || d_in >= in_depth) continue;

        int h_in_tmp = h_out + pad_h - kh;
        if (h_in_tmp % stride_h != 0) continue;
        int h_in = h_in_tmp / stride_h;
        if (h_in < 0 || h_in >= in_height) continue;

        int w_in_tmp = w_out + pad_w - kw;
        if (w_in_tmp % stride_w != 0) continue;
        int w_in = w_in_tmp / stride_w;
        if (w_in < 0 || w_in >= in_width) continue;

        int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;

        int weight_idx = ((((input_channel) * (out_channels / groups) + out_c_local) * kT + kd) * kH + kh) * kW + kw;

        sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
    }

    atomicAdd(&output[global_thread_id], sum);
    if (bias != nullptr && threadIdx.x == 0) {
        output[global_thread_id] += __ldg(&bias[c_out]);
    }
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

    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int out_channels = weight.size(1) * groups;
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    int total_output = N * out_channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_optimized_kernel", ([&] {
        transposed_conv3d_optimized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            N, in_channels, in_depth, in_height, in_width,
            out_channels, out_depth, out_height, out_width,
            kT, kH, kW,
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            output_padding[0], output_padding[1], output_padding[2],
            groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d forward function",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
