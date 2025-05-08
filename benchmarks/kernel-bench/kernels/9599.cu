#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename scalar_t>
__global__ void depthwiseConv2DKernelThreadOpt(
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
    const int total_elements) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;
    
    const int w_out = tid % out_width;
    const int h_out = (tid / out_width) % out_height;
    const int c = (tid / (out_width * out_height)) % in_channels;
    const int n = tid / (out_width * out_height * in_channels);
    
    const int input_nc_offset = (n * in_channels + c) * in_height * in_width;
    const int weight_offset = c * kernel_size * kernel_size;
    
    scalar_t sum = 0;
    
    #pragma unroll
    for (int kh = 0; kh < kernel_size; ++kh) {
        const int h_in = h_out * stride - padding + kh;
        if (h_in >= 0 && h_in < in_height) {
            const int input_h_offset = input_nc_offset + h_in * in_width;
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int w_in = w_out * stride - padding + kw;
                if (w_in >= 0 && w_in < in_width) {
                    sum += x[input_h_offset + w_in] * w[weight_offset + kh * kernel_size + kw];
                }
            }
        }
    }
    
    out[tid] = sum + b[c];
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
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());
    
    const int total_elements = batch_size * in_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_thread_opt", ([&] {
        depthwiseConv2DKernelThreadOpt<scalar_t><<<num_blocks, threads_per_block>>>(
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
            total_elements
        );
    }));
    
    return out;
}

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
        "Depthwise conv2d forward with optimized thread indexing",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}