#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Optimized depthwise 2D convolution kernel using shared memory and warp-level primitives.
template <typename scalar_t>
__global__ void depthwiseConv2DSharedKernel(
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
    extern __shared__ scalar_t shared_mem[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_channels * out_height * out_width;
    if (idx >= total) {
        return;
    }

    // Decompose idx into (n, c, h_out, w_out).
    int w_out_idx = idx % out_width;
    int tmp = idx / out_width;
    int h_out_idx = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    // Load input and weights into shared memory
    int thread_id = threadIdx.x;
    int num_threads = blockDim.x;
    for (int i = thread_id; i < kernel_size * kernel_size; i += num_threads) {
        shared_mem[i] = w[c * kernel_size * kernel_size + i];
    }
    __syncthreads();

    // Accumulate over the kernel.
    scalar_t value = 0;
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_out_idx * stride - padding + kh;
            int w_in = w_out_idx * stride - padding + kw;
            // Boundary check.
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                int w_index = kh * kernel_size + kw;
                value += x[x_index] * shared_mem[w_index];
            }
        }
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    // Add bias for this channel.
    if (threadIdx.x % warpSize == 0) {
        out[idx] = value + b[c];
    }
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
    // For depthwise conv: groups == in_channels typically.
    // Compute output dimensions.
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    const int kernel_size = weight.size(2);  // weight is (in_channels, 1, K, K)
    // Output height/width formula for convolution.
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    // Create output tensor.
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Launch kernel.
    const int total = batch_size * in_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const int shared_mem_size = kernel_size * kernel_size * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        depthwiseConv2DSharedKernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
        "Depthwise conv2d forward (handles optional bias)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}