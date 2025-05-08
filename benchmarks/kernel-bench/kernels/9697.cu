#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

// Define maximum number of elements that can be stored in constant memory
// For example, 16384 floats (64KB) or 16384 doubles (128KB) depending on hardware limits
#define MAX_CONST_SIZE 4096

// Declare constant memory arrays for float and double types
__constant__ float d_weight_const_float[MAX_CONST_SIZE];
__constant__ double d_weight_const_double[MAX_CONST_SIZE];

// Helper function to fetch from constant memory
// Specialization for float
template <typename scalar_t>
__device__ __forceinline__ scalar_t get_const_weight(int idx);

template <>
__device__ __forceinline__ float get_const_weight<float>(int idx) {
    return d_weight_const_float[idx];
}

// Specialization for double
template <>
__device__ __forceinline__ double get_const_weight<double>(int idx) {
    return d_weight_const_double[idx];
}

// CUDA kernel for depthwise 2D convolution using constant memory for the kernel weights
template <typename scalar_t>
__global__ void depthwiseConv2DKernelConstant(
    const scalar_t* __restrict__ x,
    // Weight is not passed as parameter; it is stored in constant memory
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
    const int padding
) {
    // blockIdx.z encodes combined (n, c) indices
    int bc = blockIdx.z;
    int c = bc % in_channels;
    int n = bc / in_channels;

    // Compute output spatial coordinates
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out < out_height && w_out < out_width) {
        const int batch_channel_offset = n * in_channels + c;
        scalar_t value = 0;
        // Compute top-left corner in input
        const int h_in_base = h_out * stride - padding;
        const int w_in_base = w_out * stride - padding;

        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_in_base + kh;
            if (h_in >= 0 && h_in < in_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_in_base + kw;
                    if (w_in >= 0 && w_in < in_width) {
                        int x_index = (batch_channel_offset * in_height + h_in) * in_width + w_in;
                        // Compute weight index: weights are stored consecutively per channel
                        int weight_index = (c * kernel_size * kernel_size) + (kh * kernel_size + kw);
                        value += x[x_index] * get_const_weight<scalar_t>(weight_index);
                    }
                }
            }
        }
        value += b[c];

        int out_index = (batch_channel_offset * out_height + h_out) * out_width + w_out;
        out[out_index] = value;
    }
}

// Forward implementation that copies the weight tensor into constant memory and launches the kernel

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight, // Expected shape: (in_channels, 1, kernel_size, kernel_size)
    torch::Tensor bias,
    int stride,
    int padding,
    int groups
) {
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    // Total number of weight elements
    auto weight_elements = weight.numel();
    if (weight_elements > MAX_CONST_SIZE) {
        throw std::runtime_error("Kernel weight size exceeds constant memory capacity.");
    }

    // Copy weight data to constant memory
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "copy_weight_to_constant", ([&] {
        if (std::is_same<scalar_t, float>::value) {
            cudaMemcpyToSymbol(d_weight_const_float, weight.data_ptr<scalar_t>(), weight_elements * sizeof(scalar_t));
        } else if (std::is_same<scalar_t, double>::value) {
            cudaMemcpyToSymbol(d_weight_const_double, weight.data_ptr<scalar_t>(), weight_elements * sizeof(scalar_t));
        }
    }));

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Configure grid and block dimensions
    // Using 16x16 thread block for good occupancy and coalesced access
    const dim3 threads(16, 16);
    const dim3 blocks(
         (out_width + threads.x - 1) / threads.x,
         (out_height + threads.y - 1) / threads.y,
         batch_size * in_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_constant", ([&] {
       depthwiseConv2DKernelConstant<scalar_t><<<blocks, threads>>>(
           x.data_ptr<scalar_t>(),
           bias.data_ptr<scalar_t>(),
           out.data_ptr<scalar_t>(),
           batch_size, in_channels, in_height, in_width,
           kernel_size, out_height, out_width,
           stride, padding
       );
    }));

    return out;
}

// Wrap forward_impl to handle optional bias

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int groups
) {
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
        "Depthwise conv2d forward with constant memory for weights",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
