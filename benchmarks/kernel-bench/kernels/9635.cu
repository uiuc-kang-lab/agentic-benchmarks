#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel uses __ldg() for read-only global memory accesses (x, w, b)
// and assumes that the input data is aligned to 128-bit boundaries. This can reduce
// memory latency on the NVIDIA H100.

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

    // Map thread indices to output spatial location
    const int w_out_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int h_out_idx = threadIdx.y + blockIdx.y * blockDim.y;
    const int linear_channel = blockIdx.z; // linear index: n * in_channels + c
    const int n = linear_channel / in_channels;
    const int c = linear_channel % in_channels;

    if (w_out_idx >= out_width || h_out_idx >= out_height || n >= batch_size) {
        return;
    }

    scalar_t value = 0;

    // Compute starting indices for the convolution window
    int h_in_start = h_out_idx * stride - padding;
    int w_in_start = w_out_idx * stride - padding;

    // Iterate through the kernel window
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        int h_in = h_in_start + kh;
        if (h_in < 0 || h_in >= in_height) continue;

        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            int w_in = w_in_start + kw;
            if (w_in < 0 || w_in >= in_width) continue;

            int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
            int w_index = (c * kernel_size + kh) * kernel_size + kw;
            // Use __ldg for read-only loads; assuming proper 128-bit alignment
            value += __ldg(&x[x_index]) * __ldg(&w[w_index]);
        }
    }

    // Load bias using __ldg and add
    value += __ldg(&b[c]);

    int out_idx = ((n * in_channels + c) * out_height + h_out_idx) * out_width + w_out_idx;
    out[out_idx] = value;
}

// The forward implementation for depthwise Conv2D using the optimized kernel
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
    const int kernel_size = weight.size(2);  // weight is (in_channels, 1, K, K)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Configure thread block and grid to improve coalesced memory accesses
    // Use 16x16 thread blocks for better occupancy and cache utilization
    dim3 threads(16, 16);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels
    );
    
    // Ensure we don't exceed maximum grid dimensions
    if (blocks.z > 65535) {  // Maximum z-dimension for CUDA grids
        dim3 new_blocks(
            blocks.x * ((blocks.z + 65534) / 65535),
            blocks.y,
            65535
        );
        blocks = new_blocks;
    }

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwiseConv2DKernel<scalar_t><<<blocks, threads>>>(
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

// Wrap forward_impl to handle optional bias.
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
        "Depthwise conv2d forward with __ldg-based memory accesses optimized for 128-bit alignment",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
