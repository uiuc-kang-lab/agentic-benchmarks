#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel combines improved occupancy and minimized warp divergence.
// It uses 512 threads per block and precomputes valid index ranges for the convolution window.

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
    const int padding) {

    const int w_out_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int h_out_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int c = blockIdx.z % in_channels;
    const int n = blockIdx.z / in_channels;

    if (w_out_idx >= out_width || h_out_idx >= out_height || n >= batch_size) {
        return;
    }

    scalar_t value = 0;

    // Precompute valid kernel range to avoid divergence
    const int h_in_start = h_out_idx * stride - padding;
    const int w_in_start = w_out_idx * stride - padding;

    const int kh_start = max(0, -h_in_start);
    const int kh_end = min(kernel_size, in_height - h_in_start);
    const int kw_start = max(0, -w_in_start);
    const int kw_end = min(kernel_size, in_width - w_in_start);

    // Pre-calculate base indices to reduce arithmetic in inner loops
    const int batch_channel_offset = (n * in_channels + c) * in_height;
    const int kernel_channel_offset = c * kernel_size * kernel_size;

    #pragma unroll
    for (int kh = kh_start; kh < kh_end; kh++) {
        const int h_in = h_in_start + kh;
        const int x_h_offset = (batch_channel_offset + h_in) * in_width;
        const int w_h_offset = (kernel_channel_offset + kh * kernel_size);
        
        #pragma unroll
        for (int kw = kw_start; kw < kw_end; kw++) {
            const int w_in = w_in_start + kw;
            const int x_index = x_h_offset + w_in;
            const int w_index = w_h_offset + kw;
            value += __ldg(&x[x_index]) * __ldg(&w[w_index]);
        }
    }

    value += b[c];

    const int out_idx = ((n * in_channels + c) * out_height + h_out_idx) * out_width + w_out_idx;
    out[out_idx] = value;
}

// Forward implementation that configures the kernel launch parameters using a block size of 512 threads
// (32 threads in x and 16 in y) to improve occupancy and reduce latency on the H100 GPU.

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

    // Block configuration: 32x16 gives 512 threads per block, which can better hide memory latency.
    dim3 threads(32, 16);
    // Calculate grid dimensions using ceiling division to exactly cover the output dimensions
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels
    );
    
    // Ensure we don't launch more blocks than necessary for the problem size
    blocks.x = min(blocks.x, (unsigned int)((out_width + threads.x - 1) / threads.x));
    blocks.y = min(blocks.y, (unsigned int)((out_height + threads.y - 1) / threads.y));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        optimizedDepthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
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

// Wrap forward_impl to handle optional bias input
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
        "Optimized depthwise conv2d forward",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
