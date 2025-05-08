#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Helper function to parse int or sequence of ints
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

// Custom CUDA kernel for transposed convolution
// Each thread computes one output element for the transposed convolution
// The output element at (n, c_out, h_out, w_out) is computed by gathering contributions
// from x and weight using the relation:
//    h_in = (h_out + padding - kh) / stride, if (h_out + padding - kh) is divisible by stride
//    w_in = (w_out + padding - kw) / stride
// Summation is performed over c_in and the kernel spatial dimensions.
__global__ void conv_transpose2d_kernel_custom(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr if not provided
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int H_out,
    const int W_out,
    const int kernel_size,
    const int stride,
    const int padding
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (index >= total) return;

    // Decode the flat index into (n, c_out, h_out, w_out)
    int w_out = index % W_out;
    int temp = index / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int c_out = temp % C_out;
    int n = temp / C_out;

    float sum = 0.0f;
    // Iterate over each input channel and kernel position
    for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in_prep = h_out + padding - kh;
                int w_in_prep = w_out + padding - kw;
                // Ensure that the mapping from output to input is valid (i.e. divisible by stride)
                if ((h_in_prep % stride == 0) && (w_in_prep % stride == 0)) {
                    int h_in = h_in_prep / stride;
                    int w_in = w_in_prep / stride;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in;
                        int weight_idx = c_in * (C_out * kernel_size * kernel_size) + 
                                         c_out * (kernel_size * kernel_size) + 
                                         kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    int output_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out;
    output[output_idx] = sum;
}

// Forward function called from Python
// This function extracts the dimensions, computes the output shape according to:
//  H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
//  and similarly for width. It then launches the CUDA kernel using an experimentally
//  chosen block size allowing further tuning (e.g. 32, 64, 128, 256, or 512 threads per block).

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride_obj = py::int_(1),
    py::object padding_obj = py::int_(0),
    py::object output_padding_obj = py::int_(0),
    int64_t groups = 1
) {
    auto stride_vec = parseIntArrayRef(stride_obj);
    auto padding_vec = parseIntArrayRef(padding_obj);
    auto output_padding_vec = parseIntArrayRef(output_padding_obj);

    int stride = stride_vec[0];
    int padding = padding_vec[0];
    int output_padding = output_padding_vec[0];

    // x: [N, C_in, H_in, W_in]
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // weight: [C_in, C_out, kernel_size, kernel_size] (square kernel assumed)
    int kernel_size = weight.size(2);
    int C_out = weight.size(1);

    // Calculate output spatial dimensions
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Experiment with block sizes. Adjust this value (e.g., 32, 64, 128, 256, 512) to
    // find the optimal configuration on the target NVIDIA H100 GPU.
    int block_size = 32;  // align to warp size
    int total_threads = N * C_out * H_out * W_out;
    int grid_size = (total_threads + block_size - 1) / block_size;

    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    conv_transpose2d_kernel_custom<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_size, stride, padding
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose2d with tunable block size",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
