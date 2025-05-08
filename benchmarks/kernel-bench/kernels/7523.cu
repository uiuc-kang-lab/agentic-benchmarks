#include <torch/extension.h>
#include <vector>

// Forward function implementing conv_transpose3d with CUDA streams
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Default dilation
    std::vector<int64_t> dilation = {1, 1, 1};

    // Create CUDA stream
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

    // Asynchronous memory copy to device
    auto x_device = x.to(at::kCUDA, x.options(), true);
    auto weight_device = weight.to(at::kCUDA, weight.options(), true);
    torch::Tensor bias_device;
    if (bias) {
        bias_device = bias->to(at::kCUDA, bias->options(), true);
    }

    // Set the stream for the current context
    at::cuda::CUDAStreamGuard guard(stream);

    // Call the ATen conv_transpose3d function asynchronously
    auto result = at::conv_transpose3d(
        x_device,
        weight_device,
        bias ? bias_device : torch::Tensor(),
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );

    // Synchronize the stream
    stream.synchronize();

    // Return result
    return result;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function with CUDA streams",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}