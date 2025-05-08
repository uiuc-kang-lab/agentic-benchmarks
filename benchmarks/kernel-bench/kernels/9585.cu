#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


// This kernel uses a stride loop to cover all output elements if the total workload exceeds
// the number of available threads. Each thread iterates over multiple indices by jumping
// a stride of blockDim.x * gridDim.x. This approach ensures robust handling of arbitrary
// input sizes while preserving correct boundary conditions.

template <typename scalar_t>
__global__ void depthwiseConv2DStrideLoopKernel(
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
    const int padding,
    const int total) 
{
    // Use stride loop to iterate over all output indices
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // Decompose idx into (n, c, h_out, w_out)
        int w_out = idx % out_width;
        int tmp = idx / out_width;
        int h_out = tmp % out_height;
        tmp /= out_height;
        int c = tmp % in_channels;
        int n = tmp / in_channels;

        scalar_t sum = 0;
        // Loop over the kernel window
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                // Check boundaries
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                    int w_index = ((c * 1 + 0) * kernel_size + kh) * kernel_size + kw;
                    sum += x[x_index] * w[w_index];
                }
            }
        }
        // Accumulate bias
        sum += b[c];
        out[idx] = sum;
    }
}

// Forward implementation with stride loop kernel

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

    const int kernel_size = weight.size(2);  // expecting weight shape: (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    const int total = batch_size * in_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_stride", ([&] {
        depthwiseConv2DStrideLoopKernel<scalar_t><<<blocks, threads>>>(
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
            padding,
            total
        );
    }));

    return out;
}

// Wrap forward_impl to handle optional bias in Python

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
        "Depthwise conv2d forward with stride loops for workload distribution, ensuring proper boundary handling.",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
