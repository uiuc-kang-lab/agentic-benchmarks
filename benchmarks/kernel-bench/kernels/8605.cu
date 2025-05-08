#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Maximum kernel size for manual loop unrolling
#define MAX_KERNEL_SIZE 16

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


// Custom CUDA kernel for ConvTranspose2d with manual loop unrolling
// For each output element at (n, c_out, h_out, w_out):
//   output(n, c_out, h_out, w_out) = \sum_{c_in} \sum_{i=0}^{K-1} \sum_{j=0}^{K-1}
//         [((h_out + padding - i) % stride == 0 && (w_out + padding - j) % stride == 0 && valid input)]
//         * input(n, c_in, (h_out + padding - i)/stride, (w_out + padding - j)/stride) * weight(c_in, c_out, i, j)
// If bias is provided, it is added to each output channel.

__global__ void conv_transpose2d_kernel_custom(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    const int batch_size,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int H_out,
    const int W_out,
    const int K,           // kernel size
    const int stride,
    const int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * C_out * H_out * W_out;
    if (idx >= total) return;

    // Decode linear index into 4D coordinates
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float res = 0.0f;
    
    // Manually unroll the kernel height and width loops using a fixed maximum bound
    #pragma unroll
    for (int i = 0; i < MAX_KERNEL_SIZE; i++) {
        if (i < K) {
            #pragma unroll
            for (int j = 0; j < MAX_KERNEL_SIZE; j++) {
                if (j < K) {
                    int h_in_temp = h_out + padding - i;
                    int w_in_temp = w_out + padding - j;
                    // Check if the candidate input positions align with the stride
                    if ((h_in_temp % stride == 0) && (w_in_temp % stride == 0)) {
                        int h_in = h_in_temp / stride;
                        int w_in = w_in_temp / stride;
                        // Verify that the input coordinates are valid
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            // Sum over all input channels
                            for (int c = 0; c < C_in; c++) {
                                int in_idx = n * (C_in * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
                                int weight_idx = c * (C_out * K * K) + c_out * (K * K) + i * K + j;
                                res += input[in_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        res += bias[c_out];
    }
    
    output[idx] = res;
}


// Forward function that sets up and launches the custom CUDA kernel
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object stride = py::int_(1),
    py::object padding = py::int_(0),
    py::object output_padding = py::int_(0),
    int64_t groups = 1
) {
    // Parse parameters
    auto stride_vec = parseIntArrayRef(stride);
    auto padding_vec = parseIntArrayRef(padding);
    auto output_padding_vec = parseIntArrayRef(output_padding);
    int stride_val = stride_vec[0];
    int padding_val = padding_vec[0];
    int output_padding_val = output_padding_vec[0];

    // Input tensor dimensions: [batch_size, C_in, H_in, W_in]
    int batch_size = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Weight tensor dimensions: [C_in, C_out, K, K]
    int K_val = weight.size(2);
    int C_out = weight.size(1);

    // Compute output dimensions for ConvTranspose2d
    int H_out = (H_in - 1) * stride_val - 2 * padding_val + K_val + output_padding_val;
    int W_out = (W_in - 1) * stride_val - 2 * padding_val + K_val + output_padding_val;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, C_out, H_out, W_out}, x.options());

    // Get pointers to tensor data
    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // Total number of output elements
    int total_output_elements = batch_size * C_out * H_out * W_out;

    // Set block and grid sizes
    int blockSize = 256;
    int gridSize = (total_output_elements + blockSize - 1) / blockSize;

    // Launch the kernel
    conv_transpose2d_kernel_custom<<<gridSize, blockSize>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        K_val,
        stride_val,
        padding_val
    );

    // Return the output tensor
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom ConvTranspose2d forward with manual loop unrolling",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
