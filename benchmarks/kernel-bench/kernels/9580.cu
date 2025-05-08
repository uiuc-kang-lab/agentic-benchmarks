#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NUM_STREAMS 4

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
    const int padding,
    const int batch_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_channels * out_height * out_width;
    if (idx >= total) return;

    int w_out_idx = idx % out_width;
    int tmp = idx / out_width;
    int h_out_idx = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = (tmp / in_channels) + batch_offset;

    scalar_t value = 0;
    #pragma unroll
    for (int kh = 0; kh < kernel_size; kh++) {
        #pragma unroll
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_out_idx * stride - padding + kh;
            int w_in = w_out_idx * stride - padding + kw;
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                int w_index = ((c * 1 + 0) * kernel_size + kh) * kernel_size + kw;
                value += x[x_index] * w[w_index];
            }
        }
    }
    value += b[c];
    out[idx] = value;
}

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups)
{
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int batch_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward", ([&] {
        for(int i = 0; i < NUM_STREAMS; i++) {
            int current_batch_start = i * batch_per_stream;
            int current_batch_size = std::min(batch_per_stream, 
                                            batch_size - current_batch_start);
            if(current_batch_size <= 0) continue;

            const int total = current_batch_size * in_channels * out_height * out_width;
            const int blocks = (total + threads - 1) / threads;

            depthwiseConv2DKernel<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                current_batch_size, in_channels, in_height, in_width,
                kernel_size, out_height, out_width,
                stride, padding, current_batch_start
            );
        }
    }));

    // Synchronize all streams
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
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