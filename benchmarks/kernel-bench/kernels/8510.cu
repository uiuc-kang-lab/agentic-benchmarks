#include <torch/extension.h>
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

__global__ void conv_transpose2d_strided_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int h_in,
    const int w_in,
    const int out_channels,
    const int h_out,
    const int w_out,
    const int kernel_size,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int groups,
    const int in_channels_per_group,
    const int out_channels_per_group,
    const int elements_per_thread
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * h_out * w_out;
    const int num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < total_elements; i += num_threads) {
        const int w = i % w_out;
        int tmp = i / w_out;
        const int h = tmp % h_out;
        tmp = tmp / h_out;
        const int c = tmp % out_channels;
        const int n = tmp / out_channels;

        const int g = c / out_channels_per_group;
        const int c_local = c % out_channels_per_group;

        float sum = 0.0f;

        #pragma unroll 4
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int h_in_base = h + padding_h - kh;
            
            #pragma unroll 4
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int w_in_base = w + padding_w - kw;

                if ((h_in_base % stride_h == 0) && (w_in_base % stride_w == 0)) {
                    const int h_in_idx = h_in_base / stride_h;
                    const int w_in_idx = w_in_base / stride_w;

                    if (h_in_idx >= 0 && h_in_idx < h_in && w_in_idx >= 0 && w_in_idx < w_in) {
                        #pragma unroll 4
                        for (int r = 0; r < in_channels_per_group; ++r) {
                            const int in_channel = g * in_channels_per_group + r;
                            const int input_idx = ((n * in_channels + in_channel) * h_in + h_in_idx) * w_in + w_in_idx;
                            const int weight_idx = (((g * in_channels_per_group + r) * out_channels_per_group + c_local) * kernel_size + kh) * kernel_size + kw;
                            
                            sum += __ldg(&input[input_idx]) * __ldg(&weight[weight_idx]);
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            sum += __ldg(&bias[c]);
        }

        const int output_idx = ((n * out_channels + c) * h_out + h) * w_out + w;
        output[output_idx] = sum;
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

    int stride_h = stride_vec[0];
    int stride_w = (stride_vec.size() > 1) ? stride_vec[1] : stride_h;
    int padding_h = padding_vec[0];
    int padding_w = (padding_vec.size() > 1) ? padding_vec[1] : padding_h;
    int output_padding_h = output_padding_vec[0];
    int output_padding_w = (output_padding_vec.size() > 1) ? output_padding_vec[1] : output_padding_h;

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int h_in = x.size(2);
    const int w_in = x.size(3);
    
    const int kernel_size = weight.size(2);
    int out_channels = weight.size(1) * groups;

    int h_out = (h_in - 1) * stride_h - 2 * padding_h + kernel_size + output_padding_h;
    int w_out = (w_in - 1) * stride_w - 2 * padding_w + kernel_size + output_padding_w;

    auto output_tensor = torch::zeros({batch_size, out_channels, h_out, w_out}, x.options());

    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    const int block_size = 256;
    const int elements_per_thread = 4;
    const int total_elements = batch_size * out_channels * h_out * w_out;
    const int num_blocks = (total_elements + block_size * elements_per_thread - 1) / (block_size * elements_per_thread);

    conv_transpose2d_strided_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output_tensor.data_ptr<float>(),
        batch_size,
        in_channels,
        h_in,
        w_in,
        out_channels,
        h_out,
        w_out,
        kernel_size,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        groups,
        in_channels_per_group,
        out_channels_per_group,
        elements_per_thread
    );

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with strided processing",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}