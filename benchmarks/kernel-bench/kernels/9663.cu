#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel uses a 2D thread block for spatial dimensions and a 3D grid where the z-dimension covers the combined batch and channel indices.
// This ensures that threads in a warp (which vary along threadIdx.x) access consecutive output elements in memory, improving global memory coalescing.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelCoalesced(
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

    // Use blockIdx.z to represent the combined (n, c) index.
    int bc = blockIdx.z;
    int c = bc % in_channels;
    int n = bc / in_channels;

    // Compute output spatial coordinates using 2D block indexing.
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out < out_height && w_out < out_width) {
        scalar_t value = 0;
        // Loop over the kernel window and accumulate.
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    // Compute index for input x which is in (batch, channel, height, width) layout.
                    int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                    // Weight layout: (in_channels, 1, kernel_size, kernel_size).
                    int w_index = (c * kernel_size + kh) * kernel_size + kw;
                    value += x[x_index] * w[w_index];
                }
            }
        }
        // Add bias (one per channel).
        value += b[c];

        // Write the result to the output tensor (layout: batch, channel, out_height, out_width).
        int out_index = ((n * in_channels + c) * out_height + h_out) * out_width + w_out;
        out[out_index] = value;
    }
}

// The forward implementation that sets up the 2D grid/block configuration for better coalesced memory access.

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {
    
    // Depthwise convolution: groups should equal to in_channels.
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    const int kernel_size = weight.size(2);  // weight shape: (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Use a 2D thread block so that threads in a warp (varying in x) access contiguous output memory.
    const dim3 threads(16, 16);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_coalesced", ([&] {
        depthwiseConv2DKernelCoalesced<scalar_t><<<blocks, threads>>>(
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

// Wrap the forward implementation to handle optional bias.

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
        "Depthwise conv2d forward with memory coalescing via 2D blocking",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
