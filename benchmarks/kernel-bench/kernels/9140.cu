#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int x_height, int x_width,
    int weight_height, int weight_width,
    int out_height, int out_width,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    extern __shared__ float shared_mem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int out_x = bx + tx;
    int out_y = by + ty;

    if (out_x < out_width && out_y < out_height) {
        float sum = 0.0f;
        for (int i = 0; i < weight_height; ++i) {
            for (int j = 0; j < weight_width; ++j) {
                int in_x = out_x - j * stride_w + pad_w;
                int in_y = out_y - i * stride_h + pad_h;
                if (in_x >= 0 && in_x < x_width && in_y >= 0 && in_y < x_height) {
                    sum += x[in_y * x_width + in_x] * weight[i * weight_width + j];
                }
            }
        }
        output[out_y * out_width + out_x] = sum;
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

    auto x_sizes = x.sizes();
    auto weight_sizes = weight.sizes();
    int x_height = x_sizes[2];
    int x_width = x_sizes[3];
    int weight_height = weight_sizes[2];
    int weight_width = weight_sizes[3];
    int out_height = (x_height - 1) * stride[0] - 2 * padding[0] + weight_height;
    int out_width = (x_width - 1) * stride[1] - 2 * padding[1] + weight_width;

    auto output = torch::zeros({x_sizes[0], weight_sizes[1], out_height, out_width}, x.options());

    dim3 threads(16, 16);
    dim3 blocks((out_width + threads.x - 1) / threads.x, (out_height + threads.y - 1) / threads.y);

    int shared_mem_size = threads.x * threads.y * sizeof(float);

    conv_transpose2d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        x_height, x_width,
        weight_height, weight_width,
        out_height, out_width,
        stride[0], stride[1],
        padding[0], padding[1]
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