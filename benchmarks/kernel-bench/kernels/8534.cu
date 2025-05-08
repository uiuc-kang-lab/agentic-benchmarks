#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int x_height, int x_width,
    int weight_height, int weight_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int groups
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_width && out_y < out_height) {
        float value = 0.0f;
        for (int c = 0; c < groups; ++c) {
            for (int kh = 0; kh < weight_height; ++kh) {
                for (int kw = 0; kw < weight_width; ++kw) {
                    int in_x = out_x * stride_w - padding_w + kw;
                    int in_y = out_y * stride_h - padding_h + kh;
                    if (in_x >= 0 && in_x < x_width && in_y >= 0 && in_y < x_height) {
                        value += __ldg(&x[(c * x_height + in_y) * x_width + in_x]) *
                                 __ldg(&weight[(c * weight_height + kh) * weight_width + kw]);
                    }
                }
            }
        }
        output[out_y * out_width + out_x] = value;
    }
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

    int x_height = x.size(2);
    int x_width = x.size(3);
    int weight_height = weight.size(2);
    int weight_width = weight.size(3);
    int out_height = (x_height - 1) * stride_vec[0] - 2 * padding_vec[0] + weight_height + output_padding_vec[0];
    int out_width = (x_width - 1) * stride_vec[1] - 2 * padding_vec[1] + weight_width + output_padding_vec[1];

    auto output = torch::empty({x.size(0), weight.size(1), out_height, out_width}, x.options());

    dim3 threads(16, 16);
    dim3 blocks((out_width + threads.x - 1) / threads.x, (out_height + threads.y - 1) / threads.y);

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        x_height, x_width,
        weight_height, weight_width,
        out_height, out_width,
        stride_vec[0], stride_vec[1],
        padding_vec[0], padding_vec[1],
        output_padding_vec[0], output_padding_vec[1],
        groups
    );

    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }

    return output;
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