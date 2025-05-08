#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void depthwiseConv2DWarpShuffleKernel(
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

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_elements = batch_size * in_channels * out_height * out_width;
    if (tid >= total_elements) return;

    const int w_out = tid % out_width;
    int tmp = tid / out_width;
    const int h_out = tmp % out_height;
    tmp /= out_height;
    const int c = tmp % in_channels;
    const int n = tmp / in_channels;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    scalar_t sum = 0;

    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            const int h_in = h_out * stride - padding + kh;
            const int w_in = w_out * stride - padding + kw;

            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                const int x_idx = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                const int w_idx = (c * kernel_size + kh) * kernel_size + kw;
                
                scalar_t x_val = x[x_idx];
                scalar_t w_val = w[w_idx];

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    scalar_t partial = __shfl_down_sync(FULL_MASK, x_val * w_val, offset);
                    if (lane_id < offset) {
                        sum += partial;
                    }
                }
            }
        }
    }

    if (lane_id == 0) {
        sum += b[c];
    }

    sum = __shfl_sync(FULL_MASK, sum, 0);

    if (tid < total_elements) {
        out[tid] = sum;
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
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    const int total_elements = batch_size * in_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_warp", ([&] {
        depthwiseConv2DWarpShuffleKernel<scalar_t><<<blocks, threads_per_block>>>(
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
        "Depthwise conv2d forward with warp shuffle operations",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}