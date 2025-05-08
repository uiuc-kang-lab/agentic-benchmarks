#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel is optimized for depthwise conv2d with square input and square kernel and stride==1.
// It replaces shared memory loads with warp-level primitives (__shfl_sync) to cooperatively load
// contiguous input data across a warp. Each block is a single warp (32 threads) processing a tile
// of 32 output columns for one output row. The warp cooperatively loads a contiguous block of input
// for a given kernel row using two loads per warp, and then uses __shfl_sync to access values needed
// for each output. This minimizes the overhead compared to shared memory operations while preserving
// numerical correctness.

template <typename scalar_t>
__global__ void depthwiseConv2DWarpShflKernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    int padding) {

    // Assumption: stride == 1.
    const int warp_lane = threadIdx.x;  // Each block has 32 threads (one warp).
    int out_x = blockIdx.x * 32 + warp_lane;
    int out_y = blockIdx.y;

    int nc = blockIdx.z;  // Flattened index for (n, c): n * in_channels + c
    int c = nc % in_channels;
    int n = nc / in_channels;

    scalar_t sum = 0;

    // Compute the base column index for this warp's load.
    // This is the starting global column for the tile loaded by the warp.
    int col_base = blockIdx.x * 32 - padding;

    // Loop over the kernel's vertical dimension
    for (int ky = 0; ky < kernel_size; ky++) {
        int in_y = out_y - padding + ky;
        
        // Load a contiguous block of input from the corresponding row using warp-level primitives.
        // Each warp cooperatively loads (32 + kernel_size - 1) elements from the input row.
        // r0 will hold the value for the thread's own lane for index = col_base + warp_lane.
        // Additionally, threads with lane < (kernel_size - 1) load an extra element into r1 to cover the tail.

        scalar_t r0 = 0;
        int col_idx0 = col_base + warp_lane;
        if (in_y >= 0 && in_y < in_height && col_idx0 >= 0 && col_idx0 < in_width) {
            int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + col_idx0;
            r0 = x[input_idx];
        }

        scalar_t r1 = 0;
        if (warp_lane < (kernel_size - 1)) {
            int col_idx1 = col_base + 32 + warp_lane; 
            if (in_y >= 0 && in_y < in_height && col_idx1 >= 0 && col_idx1 < in_width) {
                int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + col_idx1;
                r1 = x[input_idx];
            }
        }

        // For each column offset in the kernel window, use warp-level shuffles to obtain the required input value.
        for (int kw = 0; kw < kernel_size; kw++) {
            int global_col = col_base + warp_lane + kw;
            scalar_t in_val = 0;
            if (global_col < 0 || global_col >= in_width) {
                in_val = 0;
            } else if (warp_lane + kw < 32) {
                // Get value from r0 of the thread at lane (warp_lane + kw)
                in_val = __shfl_sync(0xFFFFFFFF, r0, warp_lane + kw);
            } else {
                int src_lane = warp_lane + kw - 32;
                // Only threads with lane index < (kernel_size - 1) loaded an extra element
                if (src_lane < (kernel_size - 1))
                    in_val = __shfl_sync(0xFFFFFFFF, r1, src_lane);
                else
                    in_val = 0;
            }
            
            // Weight index: weight is assumed to be laid out as (in_channels, 1, kernel_size, kernel_size).
            int weight_idx = (c * kernel_size + ky) * kernel_size + kw;
            scalar_t weight_val = w[weight_idx];

            sum += in_val * weight_val;
        }
    }

    sum += b[c];

    if (out_x < out_width) {
        int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}

// Forward implementation for depthwise conv2d using warp-level shuffle optimizations.
// This variant only supports stride==1.

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    TORCH_CHECK(stride == 1, "depthwise_conv2d_warp_shfl kernel only supports stride == 1");

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int kernel_size = weight.size(2); // weight shape: (in_channels, 1, k, k)

    int out_height = in_height + 2 * padding - kernel_size + 1;
    int out_width  = in_width + 2 * padding - kernel_size + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Grid configuration:
    // grid.x: Number of horizontal tiles (each tile covers 32 output columns)
    // grid.y: Each output row
    // grid.z: Each (n, c) slice (batch * channels)
    int grid_x = (out_width + 32 - 1) / 32;
    int grid_y = out_height;
    int grid_z = batch_size * in_channels;
    
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(32); // one warp per block

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_warp_shfl", ([&] {
        depthwiseConv2DWarpShflKernel<scalar_t><<<grid, block>>>(
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
            padding
        );
    }));

    return out;
}

// Wrapper function to handle optional bias

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
        "Depthwise conv2d forward with warp-level shuffle optimization (stride==1 only)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
