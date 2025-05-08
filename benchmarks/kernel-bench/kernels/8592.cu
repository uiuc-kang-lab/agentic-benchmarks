#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Utility function to parse int or sequence of ints
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

// Custom CUDA kernel for conv_transpose2d with balanced workload distribution
// Each thread computes one output element
__global__ void distributed_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * output_height * output_width;
    if (idx >= total) return;

    // Compute output indices (n, oc, h, w) from the linear index
    int w = idx % output_width;
    int tmp = idx / output_width;
    int h = tmp % output_height;
    tmp /= output_height;
    int oc = tmp % out_channels;
    int n = tmp / out_channels;

    float sum = 0.0f;

    // For each input channel, accumulate contributions
    for (int ic = 0; ic < in_channels; ic++) {
        // Iterate over kernel elements
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // In transposed convolution, the mapping is inverted:
                // Output pixel at (h, w) gets contribution from input at (i, j) where:
                // i = (h + padding - kh) / stride and j = (w + padding - kw) / stride
                int in_h_temp = h + padding - kh;
                int in_w_temp = w + padding - kw;
                // Check if the position aligns with the stride
                if (in_h_temp % stride == 0 && in_w_temp % stride == 0) {
                    int i_h = in_h_temp / stride;
                    int i_w = in_w_temp / stride;
                    if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                        // Compute linear indices for input and weight
                        int input_index = n * in_channels * input_height * input_width +
                                          ic * input_height * input_width +
                                          i_h * input_width + i_w;
                        int weight_index = ic * out_channels * kernel_size * kernel_size +
                                           oc * kernel_size * kernel_size +
                                           kh * kernel_size + kw;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
    }
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[oc];
    }
    output[idx] = sum;
}

// Host function that prepares the kernel launch
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1  // groups not supported in this custom kernel, assume 1
) {
    // Parse parameters
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);

    int stride_val = stride_vec[0];
    int padding_val = padding_vec[0];
    int output_padding_val = output_padding_vec[0];

    // Get input dimensions
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);

    // Get weight dimensions [in_channels, out_channels, kernel_size, kernel_size]
    int kernel_size = weight.size(2);  // assume square kernel
    int out_channels = weight.size(1);

    // Compute output dimensions based on conv_transpose2d formula
    int output_height = (input_height - 1) * stride_val - 2 * padding_val + kernel_size + output_padding_val;
    int output_width = (input_width - 1) * stride_val - 2 * padding_val + kernel_size + output_padding_val;

    // Allocate the output tensor
    auto options = x.options();
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, options);

    // Total number of output elements
    int total_output = batch_size * out_channels * output_height * output_width;

    // Launch parameters - using 256 threads per block
    int blockSize = 512;
    int gridSize = (total_output + blockSize - 1) / blockSize;

    // Determine bias pointer (if provided)
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch the kernel
    distributed_conv_transpose_kernel<<<gridSize, blockSize, 0, stream>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride_val,
        padding_val
    );

    // Check for kernel errors (optional error checking)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Distributed ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
