#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

__global__ void conv_transpose2d_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int h_in,
    int w_in,
    int out_channels,
    int h_out,
    int w_out,
    int kernel_size,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int groups,
    int in_channels_per_group,
    int out_channels_per_group
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * out_channels * h_out * w_out) return;

    int w_out_pos = index % w_out;
    int h_out_pos = (index / w_out) % h_out;
    int out_channel = (index / (w_out * h_out)) % out_channels;
    int batch_idx = index / (w_out * h_out * out_channels);

    int group_idx = out_channel / out_channels_per_group;
    int w_offset = 0, h_offset = 0;

    float value = 0.0f;

    #pragma unroll
    for (int kw = 0; kw < kernel_size; ++kw) {
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            int input_w = w_out_pos + padding_w - kw;
            int input_h = h_out_pos + padding_h - kh;
            
            if (input_w % stride_w == 0 && input_h % stride_h == 0) {
                input_w /= stride_w;
                input_h /= stride_h;

                if (input_w >= 0 && input_w < w_in && input_h >= 0 && input_h < h_in) {
                    for (int c = 0; c < in_channels_per_group; ++c) {
                        int in_channel = group_idx * in_channels_per_group + c;
                        int input_index = ((batch_idx * in_channels + in_channel) * h_in + input_h) * w_in + input_w;
                        int weight_index = (((out_channel * in_channels_per_group + c) * kernel_size + kh) * kernel_size + kw);
                        value += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
    }

    if (bias) {
        value += bias[out_channel];
    }

    int output_index = ((batch_idx * out_channels + out_channel) * h_out + h_out_pos) * w_out + w_out_pos;
    output[output_index] = value;
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

    int stride_h = stride_vec[0];
    int stride_w = (stride_vec.size() > 1) ? stride_vec[1] : stride_h;
    int padding_h = padding_vec[0];
    int padding_w = (padding_vec.size() > 1) ? padding_vec[1] : padding_h;

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int h_in = x.size(2);
    const int w_in = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_channels = weight.size(1) * groups;

    const int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size;
    const int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size;

    auto output = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    const int in_channels_per_group = in_channels / groups;
    const int out_channels_per_group = out_channels / groups;

    int grid_size = (batch_size * out_channels * h_out * w_out + 255) / 256;
    int block_size = 256;

    const float* input_data = x.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    conv_transpose2d_kernel_optimized<<<grid_size, block_size>>>(
        input_data, weight_data, bias_data,
        output_data,
        batch_size, in_channels, h_in, w_in,
        out_channels, h_out, w_out, kernel_size,
        stride_h, stride_w, padding_h, padding_w,
        groups, in_channels_per_group, out_channels_per_group
    );
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d Optimized Forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}