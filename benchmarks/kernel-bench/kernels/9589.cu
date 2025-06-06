#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define maximum sizes for constant memory buffers (adjust if necessary)
#define MAX_WEIGHT_SIZE 1024  // Maximum number of weights (in_channels * kernel_size * kernel_size)
#define MAX_BIAS_SIZE 256     // Maximum number of channels

// Declare constant memory buffers for weights and bias
__constant__ float const_weights[MAX_WEIGHT_SIZE];
__constant__ float const_bias[MAX_BIAS_SIZE];

// Simple depthwise 2D convolution kernel using constant memory for read-only data
// Each thread computes one output element
template <typename scalar_t>
__global__ void depthwiseConv2DKernelConst(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_channels * out_height * out_width;
    if (idx >= total) return;

    // Decompose idx into (n, c, h_out, w_out)
    int w_out = idx % out_width;
    int tmp = idx / out_width;
    int h_out = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    scalar_t sum = 0;
    // Perform the convolution over the kernel window
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out * stride - padding + kh;
            int w_in = w_out * stride - padding + kw;
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                // Compute weight index: weight expected shape (in_channels, 1, kernel_size, kernel_size)
                int w_index = ((c) * kernel_size + kh) * kernel_size + kw;
                sum += x[x_index] * const_weights[w_index];
            }
        }
    }
    
    // Add bias from constant memory
    sum += const_bias[c];
    
    out[idx] = sum;
}

// Forward implementation that copies weight and bias to constant memory and then launches the kernel
torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width  = x.size(3);

    const int kernel_size = weight.size(2);  // weight shape: (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Copy weight and bias to constant memory
    size_t weight_bytes = in_channels * kernel_size * kernel_size * sizeof(float);
    size_t bias_bytes = in_channels * sizeof(float);
    cudaMemcpyToSymbol(const_weights, weight.data_ptr<float>(), weight_bytes);
    cudaMemcpyToSymbol(const_bias, bias.data_ptr<float>(), bias_bytes);

    const int total = batch_size * in_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_const", ([&] {
        depthwiseConv2DKernelConst<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            out_height,
            out_width,
            stride,
            padding
        );
    }));

    return out;
}

// Wrap forward_impl to handle optional bias
torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int groups) {
    torch::Tensor bias;
    if (bias_obj.is_none()) {
        bias = torch::zeros({x.size(1)}, x.options());
    } else {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return forward_impl(x, weight, bias, stride, padding, groups);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Depthwise conv2d forward with constant memory for weights and bias",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
