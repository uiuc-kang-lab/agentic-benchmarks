#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper to parse an int or a sequence of ints into a 2-element vector
inline std::vector<int64_t> parseIntArrayRef(const py::object &obj) {
    std::vector<int64_t> result;
    if (py::isinstance<py::int_>(obj)) {
        int64_t val = obj.cast<int64_t>();
        result.push_back(val);
        result.push_back(val);
    } else if (py::isinstance<py::sequence>(obj)) {
        for (auto item : obj.cast<py::sequence>()) {
            result.push_back(py::cast<int64_t>(item));
        }
        if (result.size() == 1) {
            result.push_back(result[0]);
        }
    } else {
        throw std::runtime_error("Expected int or sequence of ints");
    }
    if (result.size() != 2) {
        throw std::runtime_error("Must provide exactly 2 integers for 2D operation");
    }
    return result;
}

// CUDA kernel with 2D block mapping for spatial dimensions and 3D grid for batch and channel
__global__ void conv_transposed2d_2d_kernel(
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
    // 2D mapping: each block covers a tile of output spatial positions
    int ow = blockIdx.x * blockDim.x + threadIdx.x; // output column index
    int oh = blockIdx.y * blockDim.y + threadIdx.y; // output row index

    // blockIdx.z encodes the batch and output channel indices
    int bz = blockIdx.z;
    int b = bz / out_channels;             // batch index
    int oc = bz % out_channels;            // output channel index

    if (ow >= out_w || oh >= out_h) return;

    float sum = 0.0f;

    // Identify the corresponding group for the output channel
    int group = oc / out_channels_per_group;
    int oc_mod = oc % out_channels_per_group;
    int start_ic = group * in_channels_per_group;
    int end_ic = start_ic + in_channels_per_group;

    // Loop over input channels in the group and kernel spatial dimensions
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

                int input_index = ((b * in_channels + ic) * in_h + i_h_div) * in_w + i_w_div;
                int weight_index = ((ic * out_channels_per_group + oc_mod) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_index] * weight[weight_index];
            }
        }
    }

    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }

    int output_index = ((b * out_channels + oc) * out_h + oh) * out_w + ow;
    output[output_index] = sum;
}

// Forward function wrapping the kernel launch with improved block and thread mapping
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    // Parse stride, padding, and output_padding parameters
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);

    int stride_h = stride_vec[0];
    int stride_w = stride_vec[1];
    int pad_h = padding_vec[0];
    int pad_w = padding_vec[1];
    int output_pad_h = output_padding_vec[0]; // Not used in kernel computation
    int output_pad_w = output_padding_vec[1];

    // Get input dimensions: [batch_size, in_channels, in_h, in_w]
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    // Weight dimensions: expected shape [in_channels, out_channels/groups, kernel_h, kernel_w]
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels_per_group = weight.size(1);
    int out_channels = out_channels_per_group * groups;

    // Compute output spatial dimensions
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;

    auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, x.options());

    int in_channels_per_group = in_channels / groups;

    // Setup 2D block dimensions for the spatial domain (e.g., 16x16 threads per block)
    dim3 blockDim(16, 16);
    // Grid dimensions: cover spatial positions and use gridDim.z for batch and channel
    dim3 gridDim((out_w + blockDim.x - 1) / blockDim.x,
                 (out_h + blockDim.y - 1) / blockDim.y,
                 batch_size * out_channels);

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value().contiguous();
    }

    conv_transposed2d_2d_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        (bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr),
        output.data_ptr<float>(),
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
    m.def("forward", &forward, "ConvTranspose2d forward with 2D block mapping",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
