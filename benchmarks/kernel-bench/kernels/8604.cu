#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Utility to parse int or sequence of ints
inline std::vector<int64_t> parseIntArrayRef(const py::object &obj) {
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

// Gather approach: each thread computes one output element.
// This avoids global atomics by computing complete output pixels independently.

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int B,       // batch size
    const int IC,      // input channels
    const int OC,      // output channels
    const int IH,      // input height
    const int IW,      // input width
    const int OH,      // output height
    const int OW,      // output width
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding
) {
    // Determine output pixel coordinate
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Third grid dimension encodes batch and output channel
    int b_oc = blockIdx.z;
    int b = b_oc / OC;
    int oc = b_oc % OC;

    if (ow >= OW || oh >= OH) return;

    float sum = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < IC; ++ic) {
        // Loop over kernel spatial positions
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // For conv_transpose2d, the relation is:
                // oh = ih * stride - padding + kh   and similarly for ow
                // Solve for ih: ih = (oh + padding - kh) / stride
                int ih_offset = oh + padding - kh;
                int iw_offset = ow + padding - kw;
                // Check if the offset is aligned with stride
                if ((ih_offset % stride == 0) && (iw_offset % stride == 0)) {
                    int ih = ih_offset / stride;
                    int iw = iw_offset / stride;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        // Compute linear indices
                        int input_index = ((b * IC + ic) * IH + ih) * IW + iw;
                        int weight_index = ((ic * OC + oc) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
            }
        }
    }
    int output_index = ((b * OC + oc) * OH + oh) * OW + ow;
    output[output_index] = sum;
}


// Forward function that launches the custom CUDA kernel
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
    int pad_val = padding_vec[0];
    int out_pad_val = output_padding_vec[0];

    // Get input dimensions (assumed NCHW layout)
    int B = x.size(0);
    int IC = x.size(1);
    int IH = x.size(2);
    int IW = x.size(3);

    // Get kernel dimensions and output channels (assumed weight shape: [IC, OC, K, K])
    int kernel_size = weight.size(2);  
    int OC = weight.size(1);

    // Calculate output spatial dimensions as per PyTorch's conv_transpose2d
    int OH = (IH - 1) * stride_val - 2 * pad_val + kernel_size + out_pad_val;
    int OW = (IW - 1) * stride_val - 2 * pad_val + kernel_size + out_pad_val;

    auto options = x.options();
    torch::Tensor output = torch::zeros({B, OC, OH, OW}, options);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x,
              (OH + block.y - 1) / block.y,
              B * OC);

    // Launch the kernel
    conv_transpose2d_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, IC, OC, IH, IW, OH, OW,
        kernel_size, stride_val, pad_val, out_pad_val
    );
    cudaDeviceSynchronize();

    // Add bias if provided
    if (bias.has_value() && bias.value().defined()) {
        auto b_tensor = bias.value();
        // Assumes bias shape is [OC]
        output += b_tensor.view({1, OC, 1, 1});
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ConvTranspose2d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
