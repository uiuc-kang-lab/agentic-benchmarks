#include <torch/extension.h>
#include <vector>

// Forward function implementing conv_transpose3d
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

    // Call the ATen conv_transpose3d function
    return at::conv_transpose3d(
        x,
        weight,
        bias ? *bias : torch::Tensor(),
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}