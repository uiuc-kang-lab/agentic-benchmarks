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
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int output_height,
    int output_width
) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int output_x = blockIdx.x * blockDim.x + tx;
    int output_y = blockIdx.y * blockDim.y + ty;

    if (output_x < output_width && output_y < output_height) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int input_x = output_x + kx - padding_width;
                int input_y = output_y + ky - padding_height;
                if (input_x % stride_width == 0 && input_y % stride_height == 0) {
                    input_x /= stride_width;
                    input_y /= stride_height;
                    if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height && (output_x + kx - padding_width) % stride_width == 0 && (output_y + ky - padding_height) % stride_height == 0) {
                        sum += weight[ky * kernel_width + kx] * input[input_y * input_width + input_x];
                    }
                }
            }
        }
        output[output_y * output_width + output_x] = sum;
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

    int input_height = x.size(2);
    int input_width = x.size(3);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height - 1) * stride_vec[0] - 2 * padding_vec[0] + kernel_height + output_padding_vec[0];
    int output_width = (input_width - 1) * stride_vec[1] - 2 * padding_vec[1] + kernel_width + output_padding_vec[1];

    auto output = torch::zeros({x.size(0), weight.size(1), output_height, output_width}, x.options());

    const dim3 block_size(16, 16);
    const dim3 grid_size((output_width + block_size.x - 1) / block_size.x, (output_height + block_size.y - 1) / block_size.y);

    int shared_memory_size = kernel_height * kernel_width * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv_transpose2d_kernel", ([&] {
        conv_transpose2d_kernel<<<grid_size, block_size, shared_memory_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            stride_vec[0],
            stride_vec[1],
            padding_vec[0],
            padding_vec[1],
            output_height,
            output_width
        );
    }));

    if (bias.has_value()) {
        output.add_(bias.value().view({1, -1, 1, 1}));
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