#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Optimized depthwise 2D convolution kernel using __ldg() for read-only accesses.
// Assumes that the data pointers (x, weight, bias) are allocated with 128-bit alignment to allow for
// efficient vectorized memory loads.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelOptimized(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
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
    int w_out_idx = idx % out_width;
    int tmp = idx / out_width;
    int h_out_idx = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    scalar_t value = 0;
    
    // Loop over the kernel window
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_out_idx * stride - padding + kh;
            int w_in = w_out_idx * stride - padding + kw;
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                // Compute flattened indices for input x and weight w
                int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                int w_index = ((c * 1 + 0) * kernel_size + kh) * kernel_size + kw;
                // Use __ldg() for read-only global memory loads
                value += __ldg(&x[x_index]) * __ldg(&w[w_index]);
            }
        }
    }
    // Load bias with __ldg() as well
    value += __ldg(&b[c]);
    
    out[idx] = value;
}

// Forward implementation using the optimized kernel
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
    const int in_width = x.size(3);

    const int kernel_size = weight.size(2);  // weight is (in_channels, 1, K, K)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    const int total = batch_size * in_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_opt_ldg", ([&] {
        depthwiseConv2DKernelOptimized<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), 
            weight.data_ptr<scalar_t>(), 
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            kernel_size, out_height, out_width,
            stride, padding
        );
    }));

    return out;
}

namespace py = pybind11;

// Wrap forward_impl to handle optional bias argument
torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Optimized depthwise conv2d forward using __ldg for read-only memory accesses.",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
