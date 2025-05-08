#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

// Define constant memory for weights
__constant__ float const_weight[1024]; // Adjust size as needed based on kernel size

__global__ void conv_transpose2d_kernel(const float* x, float* y, int x_width, int x_height, int y_width, int y_height, int kernel_width, int kernel_height, int stride_x, int stride_y, int padding_x, int padding_y) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < y_width && ty < y_height) {
        float value = 0.0f;
        for (int kx = 0; kx < kernel_width; ++kx) {
            for (int ky = 0; ky < kernel_height; ++ky) {
                int x_idx = tx * stride_x + kx - padding_x;
                int y_idx = ty * stride_y + ky - padding_y;
                if (x_idx >= 0 && x_idx < x_width && y_idx >= 0 && y_idx < x_height) {
                    value += x[y_idx * x_width + x_idx] * const_weight[ky * kernel_width + kx];
                }
            }
        }
        y[ty * y_width + tx] = value;
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

    int x_width = x.size(3);
    int x_height = x.size(2);
    int kernel_width = weight.size(3);
    int kernel_height = weight.size(2);
    int y_width = (x_width - 1) * stride[0] - 2 * padding[0] + kernel_width;
    int y_height = (x_height - 1) * stride[1] - 2 * padding[1] + kernel_height;

    torch::Tensor y = torch::zeros({x.size(0), weight.size(1), y_height, y_width}, x.options());

    // Copy weights to constant memory
    cudaMemcpyToSymbol(const_weight, weight.data_ptr<float>(), kernel_width * kernel_height * sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((y_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (y_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock>>>(x.data_ptr<float>(), y.data_ptr<float>(), x_width, x_height, y_width, y_height, kernel_width, kernel_height, stride[0], stride[1], padding[0], padding[1]);

    if (bias.has_value()) {
        y += bias.value().view({1, -1, 1, 1});
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride"),
          py::arg("padding"));
}