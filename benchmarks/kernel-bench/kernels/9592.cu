/*
Optimized depthwise Conv2D kernel that combines tiled shared memory loading from Kernel 2
with the simple indexing and bias addition of Kernel 1. Each block processes one (n, c) slice
of the output using a TILE_W x TILE_H tile. The kernel cooperatively loads a corresponding
input tile (including the required halo) into shared memory, then each thread computes
one output element using loop unrolling for the kernel window.
*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define tile dimensions
#define TILE_W 16
#define TILE_H 16

// Optimized depthwise convolution kernel with shared memory tiling and loop unrolling

template <typename scalar_t>
__global__ void depthwiseConv2DKernelTiledOpt(
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

    // Tile dimensions for output
    const int tile_w = TILE_W;
    const int tile_h = TILE_H;

    // Determine the top-left coordinate of the output tile
    int out_tile_x = blockIdx.x * tile_w;
    int out_tile_y = blockIdx.y * tile_h;

    // Each block handles one (n, c) slice; decode from blockIdx.z
    int nc = blockIdx.z; // flattened index for batch and channel
    int c = nc % in_channels;
    int n = nc / in_channels;

    // Compute shared memory dimensions required (includes halo for the convolution)
    int shared_width = (tile_w - 1) * stride + kernel_size;
    int shared_height = (tile_h - 1) * stride + kernel_size;

    // Allocate dynamic shared memory. Using char array casted to scalar_t pointer.
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Determine the starting coordinate in the input for this shared memory tile
    int in_tile_x = out_tile_x * stride - padding;
    int in_tile_y = out_tile_y * stride - padding;

    // Cooperative loading of input tile (with halo) into shared memory
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

    // Each thread computes one output element if within the output tile boundaries
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int out_x = out_tile_x + local_x;
    int out_y = out_tile_y + local_y;

    if (local_x < tile_w && local_y < tile_h && out_x < out_width && out_y < out_height) {
        scalar_t sum = 0;
        // Corresponding top-left in shared memory for this output element
        int sh_x = local_x * stride;
        int sh_y = local_y * stride;
        
        // Convolve over the kernel window; use loop unrolling if kernel_size is small
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int shared_idx = (sh_y + kh) * shared_width + (sh_x + kw);
                // Weight index: weight is assumed to be of shape (in_channels, 1, kernel_size, kernel_size)
                int weight_idx = (c * kernel_size + kh) * kernel_size + kw;
                sum += shmem[shared_idx] * w[weight_idx];
            }
        }
        sum += b[c]; // Add bias
        int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}

// Forward implementation using the optimized kernel

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
    const int kernel_size = weight.size(2); // weight shape: (in_channels, 1, kernel_size, kernel_size)
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Configure grid and block dimensions
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_width  + TILE_W - 1) / TILE_W,
        (out_height + TILE_H - 1) / TILE_H,
        batch_size * in_channels
    );

    // Compute shared memory size using the tensor's element size
    int shared_width = (TILE_W - 1) * stride + kernel_size;
    int shared_height = (TILE_H - 1) * stride + kernel_size;
    size_t shmem_size = shared_width * shared_height * x.element_size();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_forward_tiled_opt", ([&] {
        depthwiseConv2DKernelTiledOpt<scalar_t><<<grid, block, shmem_size>>>(
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

// Wrap forward_impl to support optional bias passed from Python (None --> zeros)

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
        "Optimized depthwise conv2d forward with shared memory tiling and loop unrolling",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
