#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Tiled depthwise conv2d kernel that uses shared memory to load an input tile corresponding
// to a tile of output elements. Each block is mapped to a spatial tile of outputs for a
// single (batch, channel) slice, with gridDim.z indexing the batch and channel.
// The thread and block indexing is organized in 2D for spatial dimensions and 3D for batch-channel,
// ensuring that threads access consecutive memory locations and the shared tile is cooperatively loaded.

template <typename scalar_t>
__global__ void depthwiseConv2DTiledKernel(
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
    int stride,
    int padding) {

    // Each block in gridDim.z corresponds to a unique (n, c) pair
    int bc = blockIdx.z;
    int c = bc % in_channels;
    int n = bc / in_channels;

    // Tile dimensions for output from block dimensions
    int tile_out_w = blockDim.x; // output tile width
    int tile_out_h = blockDim.y; // output tile height

    // Starting indices of the output tile for this block
    int out_tile_start_x = blockIdx.x * tile_out_w;
    int out_tile_start_y = blockIdx.y * tile_out_h;

    // Global output coordinates for this thread
    int out_x = out_tile_start_x + threadIdx.x;
    int out_y = out_tile_start_y + threadIdx.y;

    // Compute corresponding starting coordinates in the input
    // for the entire tile. For a given output tile,
    // the corresponding input tile starts at:
    int in_tile_start_x = out_tile_start_x * stride - padding;
    int in_tile_start_y = out_tile_start_y * stride - padding;

    // The shared memory tile needs to cover all input pixels used by this output tile.
    // Its dimensions are:
    // tile_in_w = (tile_out_w - 1) * stride + kernel_size
    // tile_in_h = (tile_out_h - 1) * stride + kernel_size
    int tile_in_w = (tile_out_w - 1) * stride + kernel_size;
    int tile_in_h = (tile_out_h - 1) * stride + kernel_size;

    // Allocate shared memory for the input tile
    extern __shared__ char smem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(smem);

    // Total number of elements in the shared input tile
    int smem_elems = tile_in_w * tile_in_h;
    int threads_per_block = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Compute base pointer for input for this (n, c) slice
    int input_base = (n * in_channels + c) * in_height * in_width;

    // Each thread cooperatively loads the shared tile
    for (int idx = tid; idx < smem_elems; idx += threads_per_block) {
        int i = idx / tile_in_w;  // row in shared tile
        int j = idx % tile_in_w;  // col in shared tile
        int in_y = in_tile_start_y + i;
        int in_x = in_tile_start_x + j;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width)
            tile[idx] = x[input_base + in_y * in_width + in_x];
        else
            tile[idx] = 0;
    }
    __syncthreads();

    // Only compute the output if within valid bounds
    if (out_y < out_height && out_x < out_width) {
        // Compute local coordinates in the shared memory tile.
        // Given out_y = out_tile_start_y + threadIdx.y, then local_y = out_y * stride - in_tile_start_y
        // Simplifies to: local_y = threadIdx.y * stride + padding
        int local_y = threadIdx.y * stride + padding;
        int local_x = threadIdx.x * stride + padding;

        scalar_t sum = 0;
        int weight_base = c * kernel_size * kernel_size;  // weight layout assumed as (in_channels, 1, kernel_size, kernel_size)
        #pragma unroll
        for (int i = 0; i < kernel_size; i++) {
            #pragma unroll
            for (int j = 0; j < kernel_size; j++) {
                int smem_index = (local_y + i) * tile_in_w + (local_x + j);
                sum += tile[smem_index] * w[weight_base + i * kernel_size + j];
            }
        }
        sum += b[c];

        int out_base = (n * in_channels + c) * out_height * out_width;
        out[out_base + out_y * out_width + out_x] = sum;
    }
}


// Forward implementation that sets up grid and block dimensions and calls the tiled kernel
// This version uses a 2D block for output tiling and a 3D grid, where gridDim.z indexes over (batch, channel).

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_height = x.size(2);
    int in_width = x.size(3);
    int kernel_size = weight.size(2);  // weight: (in_channels, 1, kernel_size, kernel_size)
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Define tile dimensions (i.e. block dimensions for output).
    const int tileW = 16;
    const int tileH = 16;
    dim3 threads(tileW, tileH);
    dim3 blocks((out_width + tileW - 1) / tileW,
                (out_height + tileH - 1) / tileH,
                batch_size * in_channels);

    // Compute shared memory size required for the input tile
    int tile_in_w = (tileW - 1) * stride + kernel_size;
    int tile_in_h = (tileH - 1) * stride + kernel_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "tiled_depthwise_conv2d_forward", ([&] {
        size_t shared_size = tile_in_w * tile_in_h * sizeof(scalar_t);
        depthwiseConv2DTiledKernel<scalar_t><<<blocks, threads, shared_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size, in_channels, in_height, in_width,
            kernel_size, out_height, out_width,
            stride, padding);
    }));

    return out;
}

// Wrap the forward implementation to handle optional bias
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
    m.def("forward", &forward_wrap, "Tiled depthwise conv2d forward with improved thread/block indexing",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("groups") = 1);
}
