#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Utility to parse int or sequence into vector
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

// __device__ helper to compute output dimensions for conv_transpose2d
__device__ inline void get_output_dims(int H_in, int W_in, int kernel_size, int stride, int padding, int output_padding, int* H_out, int* W_out) {
    // Standard formula for conv_transpose2d output dimensions
    *H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    *W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;
}

// __device__ helper to compute linear index for output tensor
__device__ inline int get_output_index(int n, int oc, int h, int w, int out_channels, int H_out, int W_out) {
    return n * (out_channels * H_out * W_out) + oc * (H_out * W_out) + h * W_out + w;
}

// __device__ helper to compute a single output element of conv_transpose2d
// Input tensor shape: [batch, in_channels, H_in, W_in]
// Weight tensor shape: [in_channels, out_channels, kernel_size, kernel_size]
// Output element computed as the sum over contributions from valid input positions.
__device__ float compute_conv_transpose_element(
    const float* input,
    const float* weight,
    int n, int oc, int h, int w,
    int in_channels, int H_in, int W_in,
    int out_channels,
    int kernel_size, int stride, int padding
) {
    float sum = 0.0f;
    // Loop over input channels and kernel spatial dimensions
    for (int c = 0; c < in_channels; c++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Compute adjusted positions
                int h_temp = h + padding - kh;
                int w_temp = w + padding - kw;
                // Check if these positions align with a valid input pixel
                if ((h_temp % stride == 0) && (w_temp % stride == 0)) {
                    int i = h_temp / stride;
                    int j = w_temp / stride;
                    if (i >= 0 && i < H_in && j >= 0 && j < W_in) {
                        int input_index = n * (in_channels * H_in * W_in) + c * (H_in * W_in) + i * W_in + j;
                        int weight_index = c * (out_channels * kernel_size * kernel_size) + oc * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
    }
    return sum;
}

// Custom conv_transpose2d kernel using modular device functions.
// Each thread computes one output element of shape [N, out_channels, H_out, W_out].
__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int H_in,
    int W_in,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int H_out, W_out;
    get_output_dims(H_in, W_in, kernel_size, stride, padding, output_padding, &H_out, &W_out);
    int total_elems = batch_size * out_channels * H_out * W_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elems) return;

    // Map linear index to (n, out_channel, h, w)
    int w = index % W_out;
    int tmp = index / W_out;
    int h = tmp % H_out;
    tmp = tmp / H_out;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float value = compute_conv_transpose_element(input, weight, n, oc, h, w,
                                                   in_channels, H_in, W_in,
                                                   out_channels, kernel_size, stride, padding);
    // If bias is provided, add it
    if (bias != nullptr) {
        value += bias[oc];
    }

    output[index] = value;
}

// Forward function called from Python via Pybind11
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    // Parse stride, padding, output_padding (assume single int for simplicity)
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);
    int stride_val = stride_vec[0];
    int padding_val = padding_vec[0];
    int output_padding_val = output_padding_vec[0];

    // Retrieve input dimensions (assume input tensor is [batch, in_channels, H_in, W_in])
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Retrieve weight dimensions (assume weight tensor is [in_channels, out_channels, kernel_size, kernel_size])
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    // Compute output dimensions based on conv_transpose2d formula
    int H_out = (H_in - 1) * stride_val - 2 * padding_val + kernel_size + output_padding_val;
    int W_out = (W_in - 1) * stride_val - 2 * padding_val + kernel_size + output_padding_val;

    auto options = x.options();
    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, options);

    int total_elems = batch_size * out_channels * H_out * W_out;
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = (bias.has_value() && bias.value().defined()) ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv_transpose2d_kernel<<<blocks, threads>>>(input_ptr, weight_ptr, bias_ptr, output_ptr,
                                                  batch_size, in_channels, H_in, W_in,
                                                  out_channels, kernel_size, stride_val, padding_val, output_padding_val);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom modular conv_transpose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
