#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to decode a flat output index into 5D tensor coordinates (n, c, d, h, w)
template <typename scalar_t>
__device__ inline void decode_output_idx(int idx, int out_channels, int out_depth, int out_height, int out_width,
                                          int &n, int &c, int &d, int &h, int &w) {
    w = idx % out_width;
    int tmp = idx / out_width;
    h = tmp % out_height;
    tmp /= out_height;
    d = tmp % out_depth;
    tmp /= out_depth;
    c = tmp % out_channels;
    n = tmp / out_channels;
}

// Device function to compute the output value at position (n, c, d, h, w)
// by gathering contributions from input and weight tensors.
template <typename scalar_t>
__device__ inline scalar_t compute_output_value(
    int n, int c, int d, int h, int w,
    int in_channels, int in_depth, int in_height, int in_width,
    int kT, int kH, int kW,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int groups, int out_channels_per_group,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight) {

    scalar_t sum = 0;
    int group = c / out_channels_per_group;
    int out_c_local = c % out_channels_per_group;
    int in_channels_per_group = in_channels / groups;

    // Loop over the input channels belonging to the current group
    for (int ic = 0; ic < in_channels_per_group; ic++) {
        int input_channel = group * in_channels_per_group + ic;
        // Loop over kernel depth
        for (int kd = 0; kd < kT; kd++) {
            int d_in_tmp = d + pad_d - kd;
            if (d_in_tmp % stride_d != 0) continue;
            int d_in = d_in_tmp / stride_d;
            if (d_in < 0 || d_in >= in_depth) continue;
            // Loop over kernel height
            for (int kh = 0; kh < kH; kh++) {
                int h_in_tmp = h + pad_h - kh;
                if (h_in_tmp % stride_h != 0) continue;
                int h_in = h_in_tmp / stride_h;
                if (h_in < 0 || h_in >= in_height) continue;
                // Loop over kernel width
                for (int kw = 0; kw < kW; kw++) {
                    int w_in_tmp = w + pad_w - kw;
                    if (w_in_tmp % stride_w != 0) continue;
                    int w_in = w_in_tmp / stride_w;
                    if (w_in < 0 || w_in >= in_width) continue;

                    // Compute flattened index for the input tensor: [N, in_channels, in_depth, in_height, in_width]
                    int input_idx = (((n * in_channels + input_channel) * in_depth + d_in) * in_height + h_in) * in_width + w_in;
                    
                    // Compute flattened index for the weight tensor:
                    // Weight shape: [in_channels, out_channels_per_group, kT, kH, kW]
                    int weight_idx = ((((input_channel) * out_channels_per_group + out_c_local) * kT + kd) * kH + kh) * kW + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    return sum;
}

// Main kernel that computes ConvTranspose3d using modular device functions
template <typename scalar_t>
__global__ void transposed_conv3d_modular_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,  // can be nullptr
    scalar_t* __restrict__ output,
    // Input dimensions
    int N, int in_channels, int in_depth, int in_height, int in_width,
    // Output dimensions
    int out_channels, int out_depth, int out_height, int out_width,
    // Kernel dimensions
    int kT, int kH, int kW,
    // Strides
    int stride_d, int stride_h, int stride_w,
    // Padding
    int pad_d, int pad_h, int pad_w,
    // Output padding (unused in computation)
    int out_pad_d, int out_pad_h, int out_pad_w,
    // Groups
    int groups,
    // Additional parameter: out_channels_per_group
    int out_channels_per_group
) {
    int total = N * out_channels * out_depth * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int n, c, d, h, w;
    decode_output_idx<scalar_t>(idx, out_channels, out_depth, out_height, out_width, n, c, d, h, w);

    scalar_t value = compute_output_value<scalar_t>(
        n, c, d, h, w,
        in_channels, in_depth, in_height, in_width,
        kT, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups, out_channels_per_group,
        input, weight);

    if (bias != nullptr) {
        value += bias[c];
    }
    output[idx] = value;
}

// Host function to launch the modular ConvTranspose3d kernel
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Ensure tensors are contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }

    // Input dimensions
    int N = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // Kernel dimensions
    int kT = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Determine out_channels using the weight tensor: [in_channels, out_channels_per_group, kT, kH, kW]
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    // Compute output dimensions based on the transposed convolution formula:
    // out_dim = (in_dim - 1) * stride - 2 * padding + kernel_size + output_padding
    int out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0];
    int out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
    int out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

    auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

    int total_elements = N * out_channels * out_depth * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_modular_kernel", ([&] {
        transposed_conv3d_modular_kernel<scalar_t><<<blocks, threads>>>(
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
            groups,
            out_channels_per_group
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward with modular device functions for improved maintainability",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}
