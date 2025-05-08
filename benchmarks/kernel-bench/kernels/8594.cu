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
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int output_height,
    const int output_width
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_c = tid % out_channels;
    const int n = (tid / out_channels) % batch_size;
    const int h_out = (tid / (out_channels * batch_size)) / output_width;
    const int w_out = (tid / (out_channels * batch_size)) % output_width;

    if (n >= batch_size || h_out >= output_height || w_out >= output_width || out_c >= out_channels)
        return;

    float sum = 0.0f;

    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            const int h_in = (h_out + padding - kh) / stride;
            const int w_in = (w_out + padding - kw) / stride;

            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                const int weight_idx = out_c * (kernel_size * kernel_size) + kh * kernel_size + kw;
                const int input_idx = ((n * in_channels) * input_height + h_in) * input_width + w_in;
                
                sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
            }
        }
    }

    output[((n * out_channels + out_c) * output_height + h_out) * output_width + w_out] = sum + (bias_ptr ? __ldg(&bias_ptr[out_c]) : 0.0f);
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

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1);

    const int output_height = (input_height - 1) * stride_vec[0] - 2 * padding_vec[0] + kernel_size + output_padding_vec[0];
    const int output_width = (input_width - 1) * stride_vec[1] - 2 * padding_vec[1] + kernel_size + output_padding_vec[1];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, x.options());

    const int threads = 256;
    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int blocks = (total_elements + threads - 1) / threads;

    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias ? bias->data_ptr<float>() : nullptr;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride_vec[0],
        padding_vec[0],
        output_padding_vec[0],
        output_height,
        output_width
    );

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
