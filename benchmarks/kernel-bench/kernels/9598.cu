#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define tile dimensions based on spatial output portion
#define TILE_W 32
#define TILE_H 8

// Optimized depthwise convolution kernel with efficient thread and block indexing.
// BlockIdx.x and BlockIdx.y index spatial tiles; BlockIdx.z indexes batch and channel.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelOptIdx(
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

    // Each block is responsible for one (n, c) slice.
    const int nc = blockIdx.z;
    const int c = nc % in_channels;
    const int n = nc / in_channels;

    // Define the tile dimensions for output
    const int tile_w = TILE_W;
    const int tile_h = TILE_H;

    // Compute the top-left index of the output tile
    const int out_tile_x = blockIdx.x * tile_w;
    const int out_tile_y = blockIdx.y * tile_h;

    // Thread indices in the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Compute dimensions of shared memory tile needed: covers the output tile plus the halo
    const int shared_width = (tile_w - 1) * stride + kernel_size;
    const int shared_height = (tile_h - 1) * stride + kernel_size;

    // Allocate dynamic shared memory
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Calculate where in the input to start loading from
    const int in_tile_x = out_tile_x * stride - padding;
    const int in_tile_y = out_tile_y * stride - padding;

    // Cooperative loading of the shared memory tile
    // Using a 2D loop with steps of block dimensions
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

    // Compute the output coordinates for this thread
    int out_x = out_tile_x + tx;
    int out_y = out_tile_y + ty;

    if (tx < tile_w && ty < tile_h && out_x < out_width && out_y < out_height) {
        scalar_t sum = 0;
        int sh_x = tx * stride;
        int sh_y = ty * stride;
        
        // Compute convolution using loop unrolling for small fixed kernel_size
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int shared_idx = (sh_y + kh) * shared_width + (sh_x + kw);
                int weight_idx = (c * kernel_size + kh) * kernel_size + kw;
                sum += shmem[shared_idx] * w[weight_idx];
            }
        }
        sum += b[c];
        int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}

// Forward implementation for the optimized kernel

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
    
    const int kernel_size = weight.size(2);  // weight expected shape: (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Set up grid: x and y dimensions for spatial tiling, z dimension for (n, c) slice
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_width  + TILE_W - 1) / TILE_W,
        (out_height + TILE_H - 1) / TILE_H,
        batch_size * in_channels
    );

    // Calculate shared memory size
    int shared_width = (TILE_W - 1) * stride + kernel_size;
    int shared_height = (TILE_H - 1) * stride + kernel_size;
    size_t shmem_size = shared_width * shared_height * x.element_size();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_opt_idx", ([&] {
        depthwiseConv2DKernelOptIdx<scalar_t><<<grid, block, shmem_size>>>(
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
        "Depthwise conv2d forward with optimized thread and block indexing mapping",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
