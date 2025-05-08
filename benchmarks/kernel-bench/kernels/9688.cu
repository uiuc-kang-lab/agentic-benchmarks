#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel uses __ldg() to optimize read-only global memory accesses and aligns memory accesses to 128-bit boundaries.
template <typename scalar_t>
__global__ void depthwiseConv2DKernelLdg(
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

    int bc = blockIdx.z;
    int c = bc % in_channels;
    int n = bc / in_channels;

    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out < out_height && w_out < out_width) {
        const int batch_channel_offset = (n * in_channels + c);
        const int w_channel_offset = c * kernel_size;
        scalar_t value = 0;

        const int h_in_base = h_out * stride - padding;
        const int w_in_base = w_out * stride - padding;

        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int h_in = h_in_base + kh;
            if (h_in >= 0 && h_in < in_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int w_in = w_in_base + kw;
                    if (w_in >= 0 && w_in < in_width) {
                        const int x_index = (batch_channel_offset * in_height + h_in) * in_width + w_in;
                        const int w_index = (w_channel_offset + kh) * kernel_size + kw;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    if constexpr (std::is_same<scalar_t, float>::value) {
        // Use fast fused multiply-add for float precision
        value = __fmaf_rn(__ldg(&x[x_index]), __ldg(&w[w_index]), value);
    } else {
        value += __ldg(&x[x_index]) * __ldg(&w[w_index]);
    }
#else
    value += __ldg(&x[x_index]) * __ldg(&w[w_index]);
#endif
                    }
                }
            }
        }
        value += __ldg(&b[c]);

        const int out_index = (batch_channel_offset * out_height + h_out) * out_width + w_out;
        out[out_index] = value;
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
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    const dim3 threads(32, 8);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_ldg", ([&] {
        depthwiseConv2DKernelLdg<scalar_t><<<blocks, threads>>>(
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
        "Depthwise conv2d forward with __ldg() optimization",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
