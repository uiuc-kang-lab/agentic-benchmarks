#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// Hybrid implementation that switches between ATen and custom kernel based on problem size
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Get input dimensions
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    // Get kernel dimensions
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    // Calculate computational complexity
    int64_t compute_size = static_cast<int64_t>(N) * C * D * H * W * kD * kH * kW;
    
    // Threshold for switching between implementations
    const int64_t COMPLEXITY_THRESHOLD = 1000000; // Tune this threshold based on hardware
    
    if (compute_size < COMPLEXITY_THRESHOLD) {
        // Use ATen implementation for small problems
        std::vector<int64_t> dilation = {1, 1, 1};
        return at::conv_transpose3d(
            input,
            weight,
            bias ? *bias : torch::Tensor(),
            stride,
            padding,
            output_padding,
            groups,
            dilation
        );
    } else {
        // Use custom gather-based kernel for large problems
        input = input.contiguous();
        weight = weight.contiguous();
        torch::Tensor bias_tensor;
        if (bias.has_value()) {
            bias_tensor = bias.value().contiguous();
        }

        // Compute output dimensions
        int out_channels = weight.size(1) * groups;
        int out_depth = (D - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0];
        int out_height = (H - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1];
        int out_width = (W - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2];

        auto output = torch::zeros({N, out_channels, out_depth, out_height, out_width}, input.options());

        int total_elements = N * out_channels * out_depth * out_height * out_width;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "transposed_conv3d_gather_kernel", ([&] {
            transposed_conv3d_gather_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.has_value() ? bias_tensor.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                N, C, D, H, W,
                out_channels, out_depth, out_height, out_width,
                kD, kH, kW,
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                output_padding[0], output_padding[1], output_padding[2],
                groups
            );
        }));

        return output;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid ConvTranspose3d forward function",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}