#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Utility to parse int or sequence of ints from Python
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

// This kernel evenly distributes the workload: each thread computes one output element (n, oc, oh, ow) using a grid-stride loop.
// For each output element, the kernel iterates over input channels and kernel spatial dimensions to accumulate contributions
// from input pixels that satisfy the transposed convolution conditions.
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int C_out,
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const int kH,
    const int kW,
    const int stride,
    const int padding,
    const int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    
    // Grid-stride loop to cover all output elements
    for (int out_idx = idx; out_idx < total; out_idx += gridDim.x * blockDim.x) {
        // Decode the linear index into (n, oc, oh, ow)
        int ow = out_idx % W_out;
        int tmp = out_idx / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int oc = tmp % C_out;
        int n = tmp / C_out;
        
        float sum = 0.0f;
        if (bias != nullptr) {
            sum = bias[oc];
        }
        
        // For conv_transpose2d, each output pixel (n, oc, oh, ow) accumulates contributions
        // from all input pixels that, when upsampled by 'stride' and shifted by kernel indices, hit (oh, ow).
        // The relation is: oh = ih * stride - padding + kh and ow = iw * stride - padding + kw.
        // Rearranging, for each kernel index (kh, kw), the corresponding input indices are:
        // ih = (oh + padding - kh) / stride, provided (oh + padding - kh) is divisible by stride
        // iw = (ow + padding - kw) / stride, provided (ow + padding - kw) is divisible by stride
        for (int ci = 0; ci < C_in; ci++) {
            for (int kh = 0; kh < kH; kh++) {
                int ih_calc = oh + padding - kh;
                if (ih_calc < 0 || (ih_calc % stride) != 0) continue;
                int ih = ih_calc / stride;
                if (ih < 0 || ih >= H_in) continue;
                for (int kw = 0; kw < kW; kw++) {
                    int iw_calc = ow + padding - kw;
                    if (iw_calc < 0 || (iw_calc % stride) != 0) continue;
                    int iw = iw_calc / stride;
                    if (iw < 0 || iw >= W_in) continue;
                    
                    int input_idx = ((n * C_in + ci) * H_in + ih) * W_in + iw;
                    int weight_idx = ((ci * C_out + oc) * kH + kh) * kW + kw;
                    sum += input[input_idx] * weight[weight_idx];
                } // kw
            } // kh
        } // ci
        
        output[out_idx] = sum;
    } // grid-stride loop
}

// The forward function allocates output, computes output dimensions based on input and kernel parameters,
// and evenly launches the kernel to compute each output element.
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

    // Use first element if provided as scalar values
    int stride = stride_vec[0];
    int padding = padding_vec[0];
    int output_padding = output_padding_vec[0];

    // Input dimensions: x is [N, C_in, H_in, W_in]
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Weight dimensions: weight is [C_in, C_out, kH, kW]
    int C_out = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    // Compute output dimensions based on conv_transpose2d formula:
    // H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
    int H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;

    auto options = x.options();
    torch::Tensor output = torch::empty({N, C_out, H_out, W_out}, options);

    int total_elements = N * C_out * H_out * W_out;
    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    const float* input_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && !bias.value().is_none()) {
        bias_ptr = bias.value().data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    conv_transpose2d_kernel<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        N, C_in, C_out, H_in, W_in, H_out, W_out, kH, kW,
        stride, padding, output_padding
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom ConvTranspose2d forward kernel",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0,
          py::arg("groups") = 1);
}
