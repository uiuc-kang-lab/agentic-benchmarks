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
    int batch_size, int in_channels, int out_channels,
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
    int out_c = blockIdx.z % out_channels;
    int batch = blockIdx.z / out_channels;

    if (out_x < out_width && out_y < out_height && batch < batch_size) {
        float value = 0.0f;
        int channels_per_group = in_channels / groups;
        int group = out_c / (out_channels / groups);
        
        for (int c = group * channels_per_group; c < (group + 1) * channels_per_group; ++c) {
            for (int kh = 0; kh < weight_height; ++kh) {
                for (int kw = 0; kw < weight_width; ++kw) {
                    int in_x = (out_x + padding_w - kw) / stride_w;
                    int in_y = (out_y + padding_h - kh) / stride_h;
                    
                    if (in_x >= 0 && in_x < x_width && in_y >= 0 && in_y < x_height &&
                        (out_x + padding_w - kw) % stride_w == 0 &&
                        (out_y + padding_h - kh) % stride_h == 0) {
                        
                        int x_idx = ((batch * in_channels + c) * x_height + in_y) * x_width + in_x;
                        int w_idx = ((out_c * channels_per_group + (c % channels_per_group)) * 
                                    weight_height + kh) * weight_width + kw;
                        
                        value += __ldg(&x[x_idx]) * __ldg(&weight[w_idx]);
                    }
                }
            }
        }
        int out_idx = ((batch * out_channels + out_c) * out_height + out_y) * out_width + out_x;
        output[out_idx] = value;
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