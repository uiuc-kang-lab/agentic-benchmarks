#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel combines block configuration for improved occupancy and grid-stride loop for better load balancing
// across threads, achieving an optimal balance between latency hiding and resource utilization.

template <typename scalar_t>
__global__ void hybridDepthwiseConv2DKernel(
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

    int total = batch_size * in_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    for (; idx < total; idx += gridSize) {
        int w_out = idx % out_width;
        int tmp = idx / out_width;
        int h_out = tmp % out_height;
        tmp /= out_height;
        int c = tmp % in_channels;
        int n = tmp / in_channels;

        scalar_t value = 0;
        int in_row_start = h_out * stride - padding;
        int in_col_start = w_out * stride - padding;

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            int in_row = in_row_start + kh;
            if (in_row >= 0 && in_row < in_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_col = in_col_start + kw;
                    if (in_col >= 0 && in_col < in_width) {
                        int x_index = ((n * in_channels + c) * in_height + in_row) * in_width + in_col;
                        int w_index = (c * kernel_size + kh) * kernel_size + kw;
                        value += x[x_index] * w[w_index];
                    }
                }
            }
        }

        value += b[c];
        int out_index = ((n * in_channels + c) * out_height + h_out) * out_width + w_out;
        out[out_index] = value;
    }
}

// Forward implementation combining both strategies
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
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    int total = batch_size * in_channels * out_height * out_width;
    dim3 threads(32, 16);  // Improved occupancy
    int blocks = (total + threads.x * threads.y - 1) / (threads.x * threads.y);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        hybridDepthwiseConv2DKernel<scalar_t><<<blocks, threads.x * threads.y>>>(
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
        "Hybrid depthwise conv2d forward with improved occupancy and grid-stride balance",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
