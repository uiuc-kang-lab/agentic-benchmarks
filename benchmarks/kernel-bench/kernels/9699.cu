#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

template <typename scalar_t>
__global__ void depthwiseConv2DKernelWarpAligned(
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

    // Each warp processes consecutive elements in the width dimension
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int wid = tid / 32;  // warp index within block
    const int lane = tid % 32; // lane index within warp
    
    // Calculate base indices for this thread block
    const int bc = blockIdx.z;
    const int c = bc % in_channels;
    const int n = bc / in_channels;
    
    // Calculate output coordinates
    const int h_out_base = blockIdx.y * (blockDim.y * 4) + threadIdx.y;
    const int w_out_base = blockIdx.x * 32 + lane; // Ensure coalesced access within warp
    
    // Pre-calculate channel offsets
    const int batch_channel_offset = (n * in_channels + c);
    const int w_channel_offset = c * kernel_size * kernel_size;
    
    // Process multiple rows per thread if needed
    for (int h_offset = 0; h_offset < 4; h_offset++) {
        const int h_out = h_out_base + h_offset * 4;
        if (h_out >= out_height || w_out_base >= out_width) continue;

        scalar_t value = 0;
        
        // Calculate input base positions
        const int h_in_base = h_out * stride - padding;
        const int w_in_base = w_out_base * stride - padding;

        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            const int h_in = h_in_base + kh;
            if (h_in >= 0 && h_in < in_height) {
                const int in_row_offset = (batch_channel_offset * in_height + h_in) * in_width;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int w_in = w_in_base + kw;
                    if (w_in >= 0 && w_in < in_width) {
                        const int x_idx = in_row_offset + w_in;
                        const int w_idx = w_channel_offset + kh * kernel_size + kw;
                        value += x[x_idx] * w[w_idx];
                    }
                }
            }
        }
        
        value += b[c];
        
        // Write output with coalesced access
        const int out_idx = (batch_channel_offset * out_height + h_out) * out_width + w_out_base;
        if (w_out_base < out_width) {
            out[out_idx] = value;
        }
    }
}

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
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Configure block and grid dimensions for warp-aligned access
    const dim3 threads(32, 4); // One warp wide, 4 rows high
    const dim3 blocks(
        (out_width + 31) / 32,  // Ensure width is covered by warps
        (out_height + 15) / 16, // Each block processes 16 rows (4 threads * 4 iterations)
        batch_size * in_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_warp_aligned", ([&] {
        depthwiseConv2DKernelWarpAligned<scalar_t><<<blocks, threads>>>(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Depthwise conv2d forward with warp-aligned memory access",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}