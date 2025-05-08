#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel precomputes valid convolution boundaries to avoid conditional checks inside inner loops,
// minimizing warp divergence by ensuring uniform control flow among threads.

template <typename scalar_t>
__global__ void depthwiseConv2DKernel(
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
    const int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_channels * out_height * out_width;
    if (idx >= total) {
        return;
    }

    // Decompose idx into (n, c, h_out, w_out).
    int w_out_idx = idx % out_width;
    int tmp = idx / out_width;
    int h_out_idx = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    scalar_t value = 0;

    // Compute the starting indices in the input tensor for the convolution window.
    int h_in_start = h_out_idx * stride - padding;
    int w_in_start = w_out_idx * stride - padding;

    // Precompute the valid range for kernel height index to avoid branching inside the inner loop
    int kh_start = (h_in_start < 0) ? -h_in_start : 0;
    int kh_end = kernel_size;
    if (h_in_start + kernel_size > in_height) {
        kh_end = in_height - h_in_start;
    }

    // Precompute the valid range for kernel width index
    int kw_start = (w_in_start < 0) ? -w_in_start : 0;
    int kw_end = kernel_size;
    if (w_in_start + kernel_size > in_width) {
        kw_end = in_width - w_in_start;
    }

    // Iterate only over the valid kernel region, avoiding conditional branches inside the loop.
    for (int kh = kh_start; kh < kh_end; kh++) {
        for (int kw = kw_start; kw < kw_end; kw++) {
            int h_in = h_in_start + kh;
            int w_in = w_in_start + kw;
            int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
            int w_index = ((c * 1 + 0) * kernel_size + kh) * kernel_size + kw;
            value += x[x_index] * w[w_index];
        }
    }

    // Add bias to the convolution result
    value += b[c];
    out[idx] = value;
}

// The forward implementation for depthwise Conv2D using the modified kernel
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

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
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

// Wrap forward_impl to handle optional bias.
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

// Pybind the module using pybind11.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Depthwise conv2d forward with minimized warp divergence",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
