#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define maximum number of elements that can be stored in constant memory for weights and bias
// Adjust these values to ensure they fit within hardware limits (typically 64KB of constant memory).
// Here we assume that the total number of weight elements (in_channels * kernel_size * kernel_size) does not exceed MAX_CONST_WEIGHT_ELEMS.

#define MAX_CONST_WEIGHT_ELEMS 1024
#define MAX_CONST_BIAS_ELEMS   1024

// Declare constant memory arrays for float and double types
__constant__ float const_weight_f[MAX_CONST_WEIGHT_ELEMS];
__constant__ float const_bias_f[MAX_CONST_BIAS_ELEMS];

__constant__ double const_weight_d[MAX_CONST_WEIGHT_ELEMS];
__constant__ double const_bias_d[MAX_CONST_BIAS_ELEMS];

// CUDA kernel that performs depthwise convolution using constant memory for the weights and bias.
// It assumes the weight tensor has been copied to constant memory beforehand.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelConstant(
    const scalar_t* __restrict__ x,
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

    // Compute combined batch and channel index using blockIdx.z
    int bc = blockIdx.z;
    int c = bc % in_channels;
    int n = bc / in_channels;

    // Compute output spatial coordinates using 2D block indexing
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out < out_height && w_out < out_width) {
        scalar_t value = 0;
        int batch_channel_offset = (n * in_channels + c);
        int h_in_base = h_out * stride - padding;
        int w_in_base = w_out * stride - padding;

        // Loop over the kernel window
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_in_base + kh;
            if (h_in >= 0 && h_in < in_height) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int w_in = w_in_base + kw;
                    if (w_in >= 0 && w_in < in_width) {
                        int x_index = (batch_channel_offset * in_height + h_in) * in_width + w_in;
                        int weight_index = (c * kernel_size + kh) * kernel_size + kw; // Layout: (in_channels, 1, K, K)
                        scalar_t kernel_val;
                        if constexpr (std::is_same<scalar_t, float>::value) {
                            kernel_val = const_weight_f[weight_index];
                        } else {
                            kernel_val = const_weight_d[weight_index];
                        }
                        value += x[x_index] * kernel_val;
                    }
                }
            }
        }

        // Add bias from constant memory
        {
            scalar_t bias_val;
            if constexpr (std::is_same<scalar_t, float>::value) {
                bias_val = const_bias_f[c];
            } else {
                bias_val = const_bias_d[c];
            }
            value += bias_val;
        }

        int out_index = (batch_channel_offset * out_height + h_out) * out_width + w_out;
        out[out_index] = value;
    }
}

// The forward implementation:
// Copies the weight and bias tensors into constant memory and launches the CUDA kernel.

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    // Retrieve input dimensions.
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);

    // For depthwise convolution, weight shape is assumed to be (in_channels, 1, kernel_size, kernel_size)
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    // Create output tensor
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Calculate the number of elements in the weight and bias tensors
    const int weight_numel = in_channels * kernel_size * kernel_size;
    const int bias_numel = in_channels;

    // Copy weight and bias data into constant memory. We assume that the weights fit within the constant memory limits.
    if (x.scalar_type() == torch::kFloat) {
        cudaMemcpyToSymbol(const_weight_f, weight.data_ptr<float>(), weight_numel * sizeof(float));
        cudaMemcpyToSymbol(const_bias_f, bias.data_ptr<float>(), bias_numel * sizeof(float));
    } else {
        cudaMemcpyToSymbol(const_weight_d, weight.data_ptr<double>(), weight_numel * sizeof(double));
        cudaMemcpyToSymbol(const_bias_d, bias.data_ptr<double>(), bias_numel * sizeof(double));
    }

    // Configure grid and block dimensions for 2D spatial mapping and combined batch/channel in grid z-dim
    const dim3 threads(16, 16);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * in_channels);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_constant", ([&] {
        depthwiseConv2DKernelConstant<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            kernel_size, out_height, out_width,
            stride, padding);
    }));

    return out;
}

// Wrap forward_impl to allow passing None as bias from Python.
// If bias is None, a zero bias tensor is created.

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
        "Depthwise conv2d forward using constant memory for weights and bias",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias") = py::none(),
        pybind11::arg("stride") = 1,
        pybind11::arg("padding") = 0,
        pybind11::arg("groups") = 1
    );
}
