#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename scalar_t>
__global__ void conv_transpose2d_aligned_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
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
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= output_width || y >= output_height || b >= batch_size)
        return;

    for (int oc = 0; oc < out_channels; ++oc) {
        scalar_t sum = 0;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int in_x = (x + padding - kw) / stride;
                    const int in_y = (y + padding - kh) / stride;
                    
                    if (in_x >= 0 && in_x < input_width && 
                        in_y >= 0 && in_y < input_height &&
                        (x + padding - kw) % stride == 0 &&
                        (y + padding - kh) % stride == 0) {
                        
                        const scalar_t input_val = __ldg(&input[
                            ((b * in_channels + ic) * input_height + in_y) * input_width + in_x
                        ]);
                        
                        const scalar_t weight_val = __ldg(&weight[
                            ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw
                        ]);
                        
                        sum += input_val * weight_val;
                    }
                }
            }
        }
        output[((b * out_channels + oc) * output_height + y) * output_width + x] = sum;
    }
}

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
    
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int input_height = x.size(2);
    const int input_width = x.size(3);
    const int out_channels = weight.size(1) * groups;
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height - 1) * stride_vec[0] - 2 * padding_vec[0] + 
                             kernel_size + output_padding_vec[0];
    const int output_width = (input_width - 1) * stride_vec[0] - 2 * padding_vec[0] + 
                            kernel_size + output_padding_vec[0];

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                             x.options());

    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y,
        batch_size
    );

    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv_transpose2d_aligned_kernel", ([&] {
        conv_transpose2d_aligned_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
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
    }));

    if (bias.has_value()) {
        output.add_(bias.value().view({1, out_channels, 1, 1}));
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