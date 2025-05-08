#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define tile dimensions for output per block
#define TILE_W 16
#define TILE_H 16

// Tiled depthwise convolution kernel using shared memory to reduce global memory accesses.
// Atomic operations are not used on global memory as each output element is computed by a unique thread.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelTiled(
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

    // Each block computes a TILE_H x TILE_W tile for one (n, c) slice
    const int tile_w = TILE_W;
    const int tile_h = TILE_H;

    // Determine the top-left output coordinate for this tile
    int out_tile_x = blockIdx.x * tile_w;
    int out_tile_y = blockIdx.y * tile_h;

    // Determine which batch and channel this block is processing
    int nc = blockIdx.z; // flattened index
    int c = nc % in_channels;
    int n = nc / in_channels;

    // Compute shared memory dimensions.
    // Shared memory tile needs to cover the output tile and the extra halo for the kernel.
    int shared_width = (tile_w - 1) * stride + kernel_size;
    int shared_height = (tile_h - 1) * stride + kernel_size;

    // Allocate dynamic shared memory (as a char array) and cast to scalar_t pointer
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Compute the starting coordinate (in the input image) for loading into shared memory
    int in_tile_x = out_tile_x * stride - padding;
    int in_tile_y = out_tile_y * stride - padding;

    // Each thread loads one or more elements of the shared memory tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int j = ty; j < shared_height; j += blockDim.y) {
        for (int i = tx; i < shared_width; i += blockDim.x) {
            int in_x = in_tile_x + i;
            int in_y = in_tile_y + j;
            scalar_t val = 0;
            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
                val = x[input_idx];
            }
            shmem[j * shared_width + i] = val;
        }
    }
    __syncthreads();

    // Each thread computes one output element from the tile if within bounds
    int out_x = out_tile_x + tx;
    int out_y = out_tile_y + ty;
    if (tx < tile_w && ty < tile_h && out_x < out_width && out_y < out_height) {
        scalar_t sum = 0;
        // The corresponding top-left index in the shared memory for this output element
        int sh_x = tx * stride;
        int sh_y = ty * stride;
        
        // Perform the convolution over the kernel window
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int shared_idx = (sh_y + kh) * shared_width + (sh_x + kw);
                // weight index: weight is of shape (in_channels, 1, kernel_size, kernel_size)
                int weight_idx = (c * kernel_size + kh) * kernel_size + kw;
                sum += shmem[shared_idx] * w[weight_idx];
            }
        }
        sum += b[c]; // Add bias

        int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}


// The forward implementation for the tiled depthwise convolution kernel.

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

    const int kernel_size = weight.size(2);  // weight is (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Define grid and block dimensions based on the output tiling
    const int tile_w = TILE_W;
    const int tile_h = TILE_H;
    dim3 block(tile_w, tile_h);
    dim3 grid(
        (out_width  + tile_w - 1) / tile_w,
        (out_height + tile_h - 1) / tile_h,
        batch_size * in_channels
    );

    // Calculate shared memory size required per block
    int shared_width = (tile_w - 1) * stride + kernel_size;
    int shared_height = (tile_h - 1) * stride + kernel_size;
    size_t shmem_size = shared_width * shared_height * sizeof(float); // works for floating point

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_tiled", ([&] {
        depthwiseConv2DKernelTiled<scalar_t><<<grid, block, shmem_size>>>(
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


// Wrap forward_impl to allow optional bias
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
        "Depthwise conv2d forward with shared memory tiling (minimal use of atomics)",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
