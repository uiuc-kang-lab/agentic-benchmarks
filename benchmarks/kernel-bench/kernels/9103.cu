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
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    for (int i = index; i < output_height * output_width; i += total_threads) {
        int out_y = i / output_width;
        int out_x = i % output_width;

        float value = 0.0f;
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int in_y = out_y * stride_h - padding_h + ky;
                int in_x = out_x * stride_w - padding_w + kx;
                if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    value += weight[ky * kernel_width + kx] * x[in_y * input_width + in_x];
                }
            }
        }
        output[out_y * output_width + out_x] = value;
    }
}

void conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    const int input_height = x.size(0);
    const int input_width = x.size(1);
    const int kernel_height = weight.size(0);
    const int kernel_width = weight.size(1);
    const int output_height = output.size(0);
    const int output_width = output.size(1);

    const int threads = 256;
    const int blocks = (output_height * output_width + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input_height, input_width,
        kernel_height, kernel_width,
        output_height, output_width,
        stride[0], stride[1],
        padding[0], padding[1]
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("stride"),
          py::arg("padding"));
}