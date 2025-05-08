#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Kernel: Each kernel launch processes a sub-batch (from batch_offset for batch_count samples).
// Each thread computes one output element for a given (n, c, h_out, w_out).

template <typename scalar_t>
__global__ void depthwiseConv2DKernelStream(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int batch_offset,
    const int batch_count,  // number of batches in this kernel launch
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_count * in_channels * out_height * out_width;
    if (idx >= total) return;

    // Decompose idx into (local batch index, channel, output height, output width)
    int w_out_idx = idx % out_width;
    int tmp = idx / out_width;
    int h_out_idx = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int local_n = tmp / in_channels; // index within this sub-batch
    int n = batch_offset + local_n; // global batch index

    scalar_t value = 0;
    // Convolve over the kernel window
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h_in = h_out_idx * stride - padding + kh;
            int w_in = w_out_idx * stride - padding + kw;
            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                int x_index = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                int w_index = ((c) * kernel_size + kh) * kernel_size + kw;  // weight shape: (in_channels, 1, K, K)
                value += x[x_index] * w[w_index];
            }
        }
    }
    value += b[c];
    int out_index = ((n * in_channels + c) * out_height + h_out_idx) * out_width + w_out_idx;
    out[out_index] = value;
}

// Forward implementation that partitions the batch dimension and pipelines kernel execution using CUDA streams

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
    const int kernel_size = weight.size(2);  // weight shape is (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Determine the number of streams to use for pipelining (e.g., 2 streams)
    const int num_streams = 2;
    // Partition the batch dimension into num_streams parts
    int sub_batch = (batch_size + num_streams - 1) / num_streams; 

    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_stream", ([&] {
        for (int i = 0; i < num_streams; i++) {
            int batch_offset = i * sub_batch;
            if (batch_offset >= batch_size) break;
            int current_batch = (sub_batch < (batch_size - batch_offset)) ? sub_batch : (batch_size - batch_offset);
            int total_threads = current_batch * in_channels * out_height * out_width;
            int blocks = (total_threads + threads - 1) / threads;
            depthwiseConv2DKernelStream<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                batch_offset,
                current_batch,
                in_channels,
                in_height,
                in_width,
                kernel_size,
                out_height,
                out_width,
                stride,
                padding);
        }
    }));

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

// Wrap forward_impl to handle the optional bias argument

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
        "Depthwise conv2d forward with stream pipelining to overlap computation and memory operations",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
