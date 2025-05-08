#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

// Kernel that processes a chunk of the batch, with an offset, allowing overlapping of computation and memory operations via CUDA streams.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelStreamed(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int n_offset,         // global batch offset for this kernel launch
    const int process_batch,    // number of batches processed in this kernel launch
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {

    // blockIdx.z covers (local batch index * in_channels).
    int bc = blockIdx.z; // ranges from 0 to process_batch * in_channels - 1
    int local_n = bc / in_channels;  // local index within the chunk
    int c = bc % in_channels;
    int n = n_offset + local_n;       // convert to global batch index

    // Compute the spatial output coordinates
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out < out_height && w_out < out_width) {
        int batch_channel_offset = n * in_channels + c;
        int w_channel_offset = c * kernel_size; // For weight indexing
        scalar_t value = 0;

        // Pre-compute base indices to reduce redundant calculations
        int h_in_base = h_out * stride - padding;
        int w_in_base = w_out * stride - padding;
        int x_base = batch_channel_offset * in_height * in_width;
        int w_base = w_channel_offset * kernel_size;

        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_in_base + kh;
            if (h_in >= 0 && h_in < in_height) {
                int x_h_offset = x_base + h_in * in_width;
                int w_h_offset = w_base + kh * kernel_size;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_in_base + kw;
                    if (w_in >= 0 && w_in < in_width) {
                        value += x[x_h_offset + w_in] * w[w_h_offset + kw];
                    }
                }
            }
        }
        value += b[c];
        int out_index = (batch_channel_offset * out_height + h_out) * out_width + w_out;
        out[out_index] = value;
    }
}


// Forward implementation that partitions the input batch and uses CUDA streams to overlap memory operations with kernel execution.

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
    const int kernel_size = weight.size(2); // weight shape: (in_channels, 1, K, K)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Use multiple CUDA streams to overlap data movement and kernel execution.
    int num_streams = std::min(batch_size, 4);  // use up to 4 streams
    int chunk_size = (batch_size + num_streams - 1) / num_streams; // ceil division

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Define kernel launch parameters
    dim3 threads(32, 8);

    // Launch kernels for each batch chunk asynchronously on its own stream
    for (int i = 0; i < num_streams; i++) {
        int n_offset = i * chunk_size;
        if (n_offset >= batch_size) break;
        int process_batch = std::min(chunk_size, batch_size - n_offset);

        // The grid 'z' dimension covers process_batch * in_channels
        dim3 blocks(
            (out_width + threads.x - 1) / threads.x,
            (out_height + threads.y - 1) / threads.y,
            process_batch * in_channels
        );

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_streamed", ([&] {
            depthwiseConv2DKernelStreamed<scalar_t><<<blocks, threads, 0, streams[i]>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                n_offset,
                process_batch,
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
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

// Wrap forward_impl to handle optional bias

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
        "Depthwise conv2d forward with overlapping computation and memory transfers using CUDA streams",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
