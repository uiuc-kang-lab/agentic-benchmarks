#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename scalar_t>
__global__ void adaptiveDepthwiseConv2DKernel(
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
    const bool use_tiled) {

    if (use_tiled) {
        const int w_out_idx = threadIdx.x + blockIdx.x * blockDim.x;
        const int h_out_idx = threadIdx.y + blockIdx.y * blockDim.y;
        const int c = blockIdx.z % in_channels;
        const int n = blockIdx.z / in_channels;

        if (w_out_idx >= out_width || h_out_idx >= out_height || n >= batch_size) {
            return;
        }

        scalar_t value = 0;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            int h_in = h_out_idx * stride - padding + kh;
            if (h_in >= 0 && h_in < in_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    int w_in = w_out_idx * stride - padding + kw;
                    if (w_in >= 0 && w_in < in_width) {
                        int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                        int w_index = (c * kernel_size + kh) * kernel_size + kw;
                        value += __ldg(&x[x_index]) * w[w_index];
                    }
                }
            }
        }
        value += b[c];
        int out_index = ((n * in_channels + c) * out_height + h_out_idx) * out_width + w_out_idx;
        out[out_index] = value;

    } else {
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
            
            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh++) {
                int h_in = h_out * stride - padding + kh;
                if (h_in >= 0 && h_in < in_height) {
                    #pragma unroll
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int w_in = w_out * stride - padding + kw;
                        if (w_in >= 0 && w_in < in_width) {
                            int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                            int w_index = (c * kernel_size + kh) * kernel_size + kw;
                            value += __ldg(&x[x_index]) * w[w_index];
                        }
                    }
                }
            }
            value += b[c];
            int out_index = ((n * in_channels + c) * out_height + h_out) * out_width + w_out;
            out[out_index] = value;
        }
    }
}

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

    const int total_elements = batch_size * in_channels * out_height * out_width;
    const bool use_tiled = total_elements <= 1048576;

    if (use_tiled) {
        dim3 threads(32, 16);
        dim3 blocks(
            (out_width + threads.x - 1) / threads.x,
            (out_height + threads.y - 1) / threads.y,
            batch_size * in_channels
        );
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
            adaptiveDepthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                batch_size, in_channels, in_height, in_width,
                kernel_size, out_height, out_width,
                stride, padding, true
            );
        }));
    } else {
        const int threads = 256;
        const int blocks = (total_elements + threads - 1) / threads;
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
            adaptiveDepthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                batch_size, in_channels, in_height, in_width,
                kernel_size, out_height, out_width,
                stride, padding, false
            );
        }));
    }

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
        "Adaptive depthwise conv2d forward",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}