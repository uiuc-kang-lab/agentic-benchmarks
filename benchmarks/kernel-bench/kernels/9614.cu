#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel uses warp-level primitives (e.g., __shfl_down_sync) to perform the reduction of the convolution sums,
// eliminating the need for shared memory for small reductions. Each warp cooperatively computes one output element.

template <typename scalar_t>
__global__ void depthwiseConv2DWarpReduceKernel(
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

    // Total number of output elements
    int total_out = batch_size * in_channels * out_height * out_width;

    // Each warp (of 32 threads) computes one output element.
    // We form warps using blockDim.x (which is fixed to 32) and blockDim.y (number of warps per block).
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (warp_id >= total_out) return;

    // Lane index within the warp
    int lane = threadIdx.x;  // 0...31

    // Decode the linear index warp_id to (n, c, out_y, out_x)
    int tmp = warp_id;
    int out_x = tmp % out_width;
    tmp /= out_width;
    int out_y = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int n = tmp / in_channels;

    // Compute the top-left corner of the input patch for this output element
    int in_origin_y = out_y * stride - padding;
    int in_origin_x = out_x * stride - padding;

    int kernel_elems = kernel_size * kernel_size;
    scalar_t partial_sum = 0;

    // Each thread in the warp processes a subset of the kernel elements (interleaved by warpSize)
    for (int idx = lane; idx < kernel_elems; idx += 32) {
        int kr = idx / kernel_size;
        int kc = idx % kernel_size;
        int in_y = in_origin_y + kr;
        int in_x = in_origin_x + kc;
        scalar_t in_val = 0;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
            in_val = x[input_idx];
        }
        int weight_idx = c * kernel_elems + idx;  // weight layout: (in_channels, 1, k, k)
        scalar_t weight_val = w[weight_idx];
        partial_sum += in_val * weight_val;
    }

    // Warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // The first lane in the warp writes the result
    if (lane == 0) {
        partial_sum += b[c];
        out[warp_id] = partial_sum;
    }
}

// Forward implementation wrapping the kernel launch

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
    const int kernel_size = weight.size(2);  // weight shape: (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width + 2 * padding - kernel_size) / stride + 1;

    int total_out = batch_size * in_channels * out_height * out_width;
    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Configure kernel launch parameters
    // We use blockDim.x = 32 (one warp per row) and choose blockDim.y as the number of warps per block.
    const int warps_per_block = 4;  // 4 warps per block (i.e., 128 threads per block)
    dim3 block(32, warps_per_block);
    int num_warps = total_out;
    int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
    dim3 grid(num_blocks);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_warp_reduce", ([&] {
        depthwiseConv2DWarpReduceKernel<scalar_t><<<grid, block>>>(
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

// Wrapper to handle optional bias from Python

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
        "Depthwise conv2d forward using warp-level reduction for small kernel operations",
        pybind11::arg("x"),
        pybind11::arg("weight"),
        pybind11::arg("bias") = py::none(),
        pybind11::arg("stride") = 1,
        pybind11::arg("padding") = 0,
        pybind11::arg("groups") = 1
    );
}
