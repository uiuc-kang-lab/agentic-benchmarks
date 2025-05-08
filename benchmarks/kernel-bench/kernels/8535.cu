#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

__constant__ float d_weight[1024];  // Adjust size as needed

inline std::vector<int64_t> parseIntArrayRef(const py::object& obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        result.push_back(obj.cast<int64_t>());
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    return result;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);

    // Copy weight to constant memory
    auto stream = at::cuda::getCurrentCUDAStream();
cudaError_t err = cudaMemcpyToSymbolAsync(d_weight, weight.data_ptr<float>(), weight.numel() * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);
if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
}
// Optionally, synchronize stream if immediate constant memory usage is required
// cudaStreamSynchronize(stream);

    return at::conv_transpose2d(
        x,
        weight,
        bias,
        stride_vec,
        padding_vec,
        output_padding_vec,
        groups,
        /* dilation */ {1, 1}
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}