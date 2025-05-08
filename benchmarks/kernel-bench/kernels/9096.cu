#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
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
    int output_width) {

    extern __shared__ float shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int output_x = bx + tx;
    int output_y = by + ty;

    if (output_x < output_width && output_y < output_height) {
        float value = 0.0f;
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int input_x = output_x - kx + padding_width;
                int input_y = output_y - ky + padding_height;
                if (input_x % stride_width == 0 && input_y % stride_height == 0) {
                    input_x /= stride_width;
                    input_y /= stride_height;
                    if (input_x >= 0 && input_x < input_width && input_y >= 0 && input_y < input_height) {
                        int input_index = input_y * input_width + input_x;
                        int weight_index = ky * kernel_width + kx;
                        value += x[input_index] * weight[weight_index];
                    }
                }
            }
        }
        output[output_y * output_width + output_x] = value;
    }
}

void conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    int input_height = x.size(0);
    int input_width = x.size(1);
    int kernel_height = weight.size(0);
    int kernel_width = weight.size(1);
    int output_height = output.size(0);
    int output_width = output.size(1);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t shared_memory_size = threadsPerBlock.x * threadsPerBlock.y * sizeof(float);

    conv_transpose2d_kernel<<<numBlocks, threadsPerBlock, shared_memory_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        output_height,
        output_width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "Conv Transpose 2D forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("stride"),
          py::arg("padding"));
}