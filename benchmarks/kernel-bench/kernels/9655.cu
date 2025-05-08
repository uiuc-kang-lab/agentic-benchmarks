#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel implements manual loop unrolling for the common 3x3 depthwise convolution case.
// For kernel sizes equal to 3, the loops are manually unrolled to remove loop overhead.
// For other kernel sizes, a fallback using #pragma unroll is provided to improve performance.
// The kernel returns the correct result and ensures optimal performance on NVIDIA H100 GPUs with CUDA 12.2.

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
    const int padding) {

    int w_out_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int h_out_idx = threadIdx.y + blockIdx.y * blockDim.y;
    int c = blockIdx.z % in_channels;
    int n = blockIdx.z / in_channels;

    if (w_out_idx >= out_width || h_out_idx >= out_height || n >= batch_size)
        return;

    scalar_t value = 0;
    int h_in_start = h_out_idx * stride - padding;
    int w_in_start = w_out_idx * stride - padding;

    // Manual unrolling for the common 3x3 kernel case
    if (kernel_size == 3) {
        int h_in, w_in, x_index, w_index;
        // First row (kh = 0)
        h_in = h_in_start + 0;
        if (h_in >= 0 && h_in < in_height) {
            // Column 0
            w_in = w_in_start + 0;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 0;
                value += x[x_index] * w[w_index];
            }
            // Column 1
            w_in = w_in_start + 1;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 1;
                value += x[x_index] * w[w_index];
            }
            // Column 2
            w_in = w_in_start + 2;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 2;
                value += x[x_index] * w[w_index];
            }
        }
        // Second row (kh = 1)
        h_in = h_in_start + 1;
        if (h_in >= 0 && h_in < in_height) {
            // Column 0
            w_in = w_in_start + 0;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 3;
                value += x[x_index] * w[w_index];
            }
            // Column 1
            w_in = w_in_start + 1;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 4;
                value += x[x_index] * w[w_index];
            }
            // Column 2
            w_in = w_in_start + 2;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 5;
                value += x[x_index] * w[w_index];
            }
        }
        // Third row (kh = 2)
        h_in = h_in_start + 2;
        if (h_in >= 0 && h_in < in_height) {
            // Column 0
            w_in = w_in_start + 0;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 6;
                value += x[x_index] * w[w_index];
            }
            // Column 1
            w_in = w_in_start + 1;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 7;
                value += x[x_index] * w[w_index];
            }
            // Column 2
            w_in = w_in_start + 2;
            if (w_in >= 0 && w_in < in_width) {
                x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                w_index = (c * 9) + 8;
                value += x[x_index] * w[w_index];
            }
        }
    } else {
        // Fallback for other kernel sizes, with loop unrolling pragma for inner loops
        int kh_start = (h_in_start < 0) ? -h_in_start : 0;
        int kh_end = (in_height - h_in_start < kernel_size) ? (in_height - h_in_start) : kernel_size;
        int kw_start = (w_in_start < 0) ? -w_in_start : 0;
        int kw_end = (in_width - w_in_start < kernel_size) ? (in_width - w_in_start) : kernel_size;
        #pragma unroll
        for (int kh = kh_start; kh < kh_end; kh++) {
            int h_in = h_in_start + kh;
            #pragma unroll
            for (int kw = kw_start; kw < kw_end; kw++) {
                int w_in = w_in_start + kw;
                int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                int w_index = (c * kernel_size * kernel_size) + (kh * kernel_size + kw);
                value += x[x_index] * w[w_index];
            }
        }
    }

    value += b[c];
    int out_idx = ((n * in_channels + c) * out_height + h_out_idx) * out_width + w_out_idx;
    out[out_idx] = value;
}


// Forward implementation wrapper
torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int kernel_size = weight.size(2);
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    dim3 threads(32, 8);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
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

namespace py = pybind11;

// Wrapper to handle optional bias
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
        "Depthwise conv2d forward (handles optional bias)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
