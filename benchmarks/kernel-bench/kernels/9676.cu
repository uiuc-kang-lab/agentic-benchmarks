#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel uses shared memory to cache weights for each channel, reducing global memory access latency.
template <typename scalar_t>
__global__ void depthwiseConv2DKernelSharedMemory(
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

    // Shared memory for kernel weights
    extern __shared__ scalar_t shared_weights[];
    int bc = blockIdx.z;
    int c = bc % in_channels;
    int n = bc / in_channels;

    // Load weights into shared memory
    if (threadIdx.y == 0 && threadIdx.x < kernel_size * kernel_size) {
        shared_weights[threadIdx.x] = w[c * kernel_size * kernel_size + threadIdx.x];
    }
    __syncthreads();

    // Compute output spatial coordinates
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out < out_height && w_out < out_width) {
        const int batch_channel_offset = (n * in_channels + c);
        scalar_t value = 0;

        // Calculate input base position
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
                        const int w_index = kh * kernel_size + kw;
                        value += x[x_index] * shared_weights[w_index];
                    }
                }
            }
        }
        value += b[c];

        const int out_index = (batch_channel_offset * out_height + h_out) * out_width + w_out;
        out[out_index] = value;
    }
}

// Forward implementation using the shared memory kernel
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

    // Block dimensions: 32x8 threads per block
    const dim3 threads(32, 8);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels
    );

    // Shared memory size for kernel weights
    const int shared_memory_size = kernel_size * kernel_size * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_shared_memory", ([&] {
        depthwiseConv2DKernelSharedMemory<scalar_t><<<blocks, threads, shared_memory_size>>>(
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

// Wrap the forward implementation to handle optional bias.
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
        "Depthwise conv2d forward with shared memory optimization",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}