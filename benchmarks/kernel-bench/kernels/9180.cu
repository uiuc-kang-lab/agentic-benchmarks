#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int output_height, int output_width,
    int stride_height, int stride_width,
    int padding_height, int padding_width
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < output_width && out_y < output_height) {
        float value = 0.0f;
        
        // Calculate the input position that corresponds to this output position
        int in_y_start = (out_y + padding_height) / stride_height;
        int in_x_start = (out_x + padding_width) / stride_width;
        
        // Iterate through all possible contributing input positions
        for (int k_y = 0; k_y < kernel_height; ++k_y) {
            int in_y = in_y_start - k_y;
            if (in_y >= 0 && in_y < input_height) {
                for (int k_x = 0; k_x < kernel_width; ++k_x) {
                    int in_x = in_x_start - k_x;
                    if (in_x >= 0 && in_x < input_width) {
                        // Check if this input position contributes to the output
                        if ((out_y + padding_height - k_y) % stride_height == 0 &&
                            (out_x + padding_width - k_x) % stride_width == 0) {
                            // Use flipped weights for transposed convolution
                            int w_y = kernel_height - 1 - k_y;
                            int w_x = kernel_width - 1 - k_x;
                            value += x[in_y * input_width + in_x] * 
                                    weight[w_y * kernel_width + w_x];
                        }
                    }
                }
            }
        }
        output[out_y * output_width + out_x] = value;
    }
}

torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    c10::optional<torch::Tensor> bias = c10::nullopt;
    if (!bias_obj.is_none()) {
        bias = bias_obj.cast<torch::Tensor>();
    }

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    int input_height = x.size(2);
    int input_width = x.size(3);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    int stride_height = stride[0];
    int stride_width = stride[1];
    int padding_height = padding[0];
    int padding_width = padding[1];
    int output_height = (input_height - 1) * stride_height - 2 * padding_height + kernel_height;
    int output_width = (input_width - 1) * stride_width - 2 * padding_width + kernel_width;

    auto output = torch::zeros({x.size(0), weight.size(1), output_height, output_width}, options);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input_height, input_width,
        kernel_height, kernel_width,
        output_height, output_width,
        stride_height, stride_width,
        padding_height, padding_width
    );

    if (bias.has_value()) {
        output += bias.value().view({1, -1, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}