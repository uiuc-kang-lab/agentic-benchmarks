#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constant memory for small read-only data
__constant__ float const_bias[1024]; // Adjust size based on expected max size

// Function to copy bias to constant memory
void copy_bias_to_constant(const float* bias, int size) {
    cudaMemcpyToSymbol(const_bias, bias, size * sizeof(float));
}

// Kernel using constant memory for bias
template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride,
    const int padding,
    const int groups,
    const int dilation,
    const int out_height,
    const int out_width
) {
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int ow = idx % out_width;
    int tmp = idx / out_width;
    int oh = tmp % out_height;
    tmp /= out_height;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    const int out_channels_per_group = out_channels / groups;
    scalar_t val = compute_conv_value<scalar_t>(
        input, weight, b, oc, oh, ow,
        in_channels, in_height, in_width,
        kernel_h, kernel_w, stride, padding,
        dilation, groups, out_channels_per_group
    );

    // Use constant memory for bias
    if (oc < 1024) { // Ensure within bounds
        val += const_bias[oc];
    }

    output[idx] = val;
}

// Forward function with constant memory bias copy
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation = 1
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() == out_channels, "Bias must have out_channels elements");
        TORCH_CHECK(bias->device().is_cuda(), "Bias must be a CUDA tensor");
        // Copy bias to constant memory
        copy_bias_to_constant(bias->data_ptr<float>(), out_channels);
    }

    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, x.options());

    const int total_elements = output.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_cuda", ([&] {
        conv_transpose2d_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
            dilation,
            out_height,
            out_width
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 2D convolution (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation") = 1);
}
