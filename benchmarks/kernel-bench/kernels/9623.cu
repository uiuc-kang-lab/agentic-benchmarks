#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Optimized depthwise 2D convolution kernel using better thread-block mapping.
template <typename scalar_t>
__global__ void optimizedDepthwiseConv2DKernel(
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
    int n = blockIdx.x;
    int c = blockIdx.y;
    int h_out_idx = blockIdx.z * blockDim.y + threadIdx.y;
    int w_out_idx = threadIdx.x;

    if (h_out_idx >= out_height || w_out_idx >= out_width) {
        return;
    }

    // Accumulate over the kernel.
    scalar_t value = 0;
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_out_idx * stride - padding + kh;
            int w_in = w_out_idx * stride - padding + kw;
            // Boundary check.
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                int w_index = ((c * 1 + 0) * kernel_size + kh) * kernel_size + kw;
                value += x[x_index] * w[w_index];
            }
        }
    }
    // Add bias for this channel.
    value += b[c];

    // Write to output.
    int out_index = ((n * in_channels + c) * out_height + h_out_idx) * out_width + w_out_idx;
    out[out_index] = value;
}

// The actual implementation of depthwise Conv2D in CUDA.
torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups)
{
    // Compute output dimensions.
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    const int kernel_size = weight.size(2);  // weight is (in_channels, 1, K, K)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    // Create output tensor.
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    dim3 block_dim(16, 16);  // threads
    dim3 grid_dim(batch_size, in_channels, (out_height + block_dim.y - 1) / block_dim.y);

    // Launch kernel.
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_depthwise_conv2d_forward", ([&] {
        optimizedDepthwiseConv2DKernel<scalar_t><<<grid_dim, block_dim>>>(
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

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    py::object bias_obj,
    int stride,
    int padding,
    int groups)
{
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
        "Optimized Depthwise conv2d forward",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}