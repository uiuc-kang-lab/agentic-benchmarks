#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

__global__ void conv_transpose2d_kernel(const float* __restrict__ x,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int x_w, int x_h, int out_w, int out_h,
                                        int kernel_w, int kernel_h,
                                        int stride_w, int stride_h,
                                        int pad_w, int pad_h) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_n = blockIdx.z;  // As we want each block in z to handle a batch element

    if (out_x < out_w && out_y < out_h) {
        float value = 0.0f;
        for (int c = 0; c < blockDim.z; ++c) {  // Modify to properly traverse channel
            for (int j = 0; j < kernel_h; ++j) {
                for (int i = 0; i < kernel_w; ++i) {
                    int in_x = out_x * stride_w - pad_w + i;
                    int in_y = out_y * stride_h - pad_h + j;
                    if (in_x >= 0 && in_x < x_w && in_y >= 0 && in_y < x_h) {
                        value += x[(out_n * x_h * x_w) + (in_y * x_w) + in_x] *
                                 weight[(c * kernel_h * kernel_w) + (j * kernel_w) + i];
                    }
                }
            }
        }
        if (bias != nullptr) {
            value += bias[out_n];
        }
        output[(out_n * out_h * out_w) + (out_y * out_w) + out_x] = value;
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

    const auto x_size = x.sizes();
    const auto weight_size = weight.sizes();
    int batch_size = x_size[0];
    int channels = weight_size[0];
    int x_h = x_size[2];
    int x_w = x_size[3];
    int kernel_h = weight_size[2];
    int kernel_w = weight_size[3];
    int out_h = (x_h - 1) * stride[0] - 2 * padding[0] + kernel_h;
    int out_w = (x_w - 1) * stride[1] - 2 * padding[1] + kernel_w;

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, channels, out_h, out_w}, options);

    dim3 block_dim(16, 16, channels); // Using channels as depth of the block
    dim3 grid_dim((out_w + block_dim.x - 1) / block_dim.x,
                  (out_h + block_dim.y - 1) / block_dim.y,
                  batch_size);

    conv_transpose2d_kernel<<<grid_dim, block_dim>>>(x.data_ptr<float>(),
                                                     weight.data_ptr<float>(),
                                                     bias ? bias->data_ptr<float>() : nullptr,
                                                     output.data_ptr<float>(),
                                                     x_w, x_h, out_w, out_h, kernel_w, kernel_h,
                                                     stride[0], stride[1], padding[0], padding[1]);

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