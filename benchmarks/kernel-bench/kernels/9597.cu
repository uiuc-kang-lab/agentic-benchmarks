#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename scalar_t>
__global__ void depthwiseConv2DKernel1D(
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
    const int total_elements)
{
    // Use 1D indexing for simplified thread mapping
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stride loop to handle cases where total_elements > number of threads
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        // Convert linear index to n,c,h,w coordinates
        const int w_out = idx % out_width;
        const int h_out = (idx / out_width) % out_height;
        const int c = (idx / (out_width * out_height)) % in_channels;
        const int n = idx / (out_width * out_height * in_channels);

        scalar_t sum = 0;

        // Compute input starting position for convolution
        const int h_in_start = h_out * stride - padding;
        const int w_in_start = w_out * stride - padding;
        
        // Optimize inner loops for small kernel sizes
        #pragma unroll 3
        for (int kh = 0; kh < kernel_size; ++kh) {
            const int h_in = h_in_start + kh;
            
            #pragma unroll 3
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int w_in = w_in_start + kw;
                
                if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                    const int in_idx = ((n * in_channels + c) * in_height + h_in) * in_width + w_in;
                    const int w_idx = (c * kernel_size + kh) * kernel_size + kw;
                    sum += x[in_idx] * w[w_idx];
                }
            }
        }
        
        // Add bias and write output
        sum += b[c];
        out[idx] = sum;
    }
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

    // Calculate total number of output elements
    const int total_elements = batch_size * in_channels * out_height * out_width;
    
    // Use 1D grid and block configuration
    const int thread_count = 256;
    const int block_count = std::min(65535, (total_elements + thread_count - 1) / thread_count);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_1d", ([&] {
        depthwiseConv2DKernel1D<scalar_t><<<block_count, thread_count>>>(
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
            padding,
            total_elements
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

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Depthwise conv2d forward with 1D indexing",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}