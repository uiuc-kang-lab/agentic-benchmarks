#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define WARPS_PER_BLOCK 8

template <typename scalar_t>
__global__ void depthwiseConv2DWarpShuffleKernel(
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

    // Calculate indices
    const int nc = blockIdx.z;
    const int c = nc % in_channels;
    const int n = nc / in_channels;

    // Calculate output position
    const int out_x = blockIdx.x * BLOCK_SIZE_X + (threadIdx.x % BLOCK_SIZE_X);
    const int out_y = blockIdx.y * BLOCK_SIZE_Y + (threadIdx.x / BLOCK_SIZE_X);
    
    // Warp and lane identification
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Load kernel weights into registers using warp shuffling
    scalar_t weight_cache[9];  // Assuming max kernel size of 3x3
    if (lane_id < kernel_size * kernel_size) {
        weight_cache[lane_id] = w[c * kernel_size * kernel_size + lane_id];
    }
    
    // Make weights available to all threads in the warp through shuffling
    #pragma unroll
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        weight_cache[i] = __shfl_sync(0xffffffff, weight_cache[i], i % WARP_SIZE);
    }

    if (out_x < out_width && out_y < out_height) {
        scalar_t sum = 0;
        
        // Calculate input starting position
        const int in_x_base = out_x * stride - padding;
        const int in_y_base = out_y * stride - padding;

        // Compute convolution
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            const int in_y = in_y_base + ky;
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                const int in_x = in_x_base + kx;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    const int in_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
                    const int w_idx = ky * kernel_size + kx;
                    sum += x[in_idx] * weight_cache[w_idx];
                }
            }
        }

        // Add bias
        sum += b[c];

        // Write output
        const int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
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

    // Configure grid and block dimensions
    dim3 block(WARP_SIZE * WARPS_PER_BLOCK);  // Multiple warps per block
    dim3 grid(
        (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        batch_size * in_channels
    );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_warp_shuffle", ([&] {
        depthwiseConv2DWarpShuffleKernel<scalar_t><<<grid, block>>>(
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
            padding
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

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",
          &forward_wrap,
          "Depthwise conv2d forward with warp shuffle operations",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("groups") = 1);
}