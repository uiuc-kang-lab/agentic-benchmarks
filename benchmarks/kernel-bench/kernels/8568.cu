#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Device function to compute a single output element
__device__ float compute_output_element(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int n, int oc, int oh, int ow,
    int in_channels, int in_h, int in_w,
    int out_channels_per_group, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int in_channels_per_group, int group
) {
    float sum = 0.0f;
    int start_ic = group * in_channels_per_group;
    int end_ic = start_ic + in_channels_per_group;

    for (int ic = start_ic; ic < end_ic; ic++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            int i_h = oh + pad_h - kh;
            if (i_h % stride_h != 0) continue;
            int i_h_div = i_h / stride_h;
            if (i_h_div < 0 || i_h_div >= in_h) continue;
            for (int kw = 0; kw < kernel_w; kw++) {
                int i_w = ow + pad_w - kw;
                if (i_w % stride_w != 0) continue;
                int i_w_div = i_w / stride_w;
                if (i_w_div < 0 || i_w_div >= in_w) continue;

                int input_index = ((n * in_channels + ic) * in_h + i_h_div) * in_w + i_w_div;
                int weight_index = ((ic) * out_channels_per_group + (oc % out_channels_per_group)) * (kernel_h * kernel_w) + (kernel_h - 1 - kh) * kernel_w + (kernel_w - 1 - kw);

                sum += input[input_index] * weight[weight_index];
            }
        }
    }

    return sum;
}

// Kernel function using the device function
__global__ void conv_transposed2d_modular_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int in_channels_per_group,
    int out_channels_per_group
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_h * out_w;

    while (idx < total) {
        int ow = idx % out_w;
        int temp = idx / out_w;
        int oh = temp % out_h;
        temp /= out_h;
        int oc = temp % out_channels;
        int n = temp / out_channels;

        int group = oc / out_channels_per_group;
        float sum = compute_output_element(
            input, weight, n, oc, oh, ow,
            in_channels, in_h, in_w,
            out_channels_per_group, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w,
            in_channels_per_group, group
        );

        if (bias != nullptr) {
            sum += bias[oc];
        }

        output[idx] = sum;
        idx += blockDim.x * gridDim.x;
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

    int stride_h = stride_vec[0];
    int stride_w = stride_vec[1];
    int pad_h = padding_vec[0];
    int pad_w = padding_vec[1];
    int output_pad_h = output_padding_vec[0];
    int output_pad_w = output_padding_vec[1];

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    int in_channels_per_group = in_channels / groups;

    int total = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value().contiguous();
    }

    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias_tensor.defined()) ? bias_tensor.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv_transposed2d_modular_kernel<<<blocks, threads>>>(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        in_channels_per_group,
        out_channels_per_group
    );
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward with modular device functions",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
