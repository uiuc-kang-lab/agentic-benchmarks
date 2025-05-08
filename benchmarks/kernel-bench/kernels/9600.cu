#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Define tile dimensions for output computation
#define TILE_W 16
#define TILE_H 16

// This kernel leverages shared memory to load both the input tile (with halo) and the per-channel kernel weights.
// By loading the frequently reused data into shared memory, we reduce global memory latency.
// Each block processes a tile of the output for one (n, c) slice. The kernel weights are loaded once per block to shared memory,
// ensuring no race conditions and efficient reuse.

template <typename scalar_t>
__global__ void depthwiseConv2DKernelShared(
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

    // Determine which (n, c) slice this block handles
    const int nc = blockIdx.z;  // flattened index: nc = n * in_channels + c
    const int c = nc % in_channels;
    const int n = nc / in_channels;

    // Tile dimensions for output
    const int tile_w = TILE_W;
    const int tile_h = TILE_H;

    // Compute the top-left coordinates for the output tile
    const int out_tile_x = blockIdx.x * tile_w;
    const int out_tile_y = blockIdx.y * tile_h;

    // Compute shared memory dimensions for the input tile (including halo)
    const int s_width = (tile_w - 1) * stride + kernel_size;
    const int s_height = (tile_h - 1) * stride + kernel_size;
    const int tile_area = s_width * s_height;

    // Allocate dynamic shared memory. The first part is for the input tile,
    // the second part is for the kernel weights (kernel_size x kernel_size elements).
    extern __shared__ char smem[];
    scalar_t* sh_input = reinterpret_cast<scalar_t*>(smem);
    scalar_t* sh_kernel = sh_input + tile_area;

    // Determine starting coordinates in input for loading the tile
    const int in_tile_x = out_tile_x * stride - padding;
    const int in_tile_y = out_tile_y * stride - padding;

    // Total number of threads in the block
    const int total_threads = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Cooperative loading of the input tile into shared memory
    // Each thread loads multiple elements if necessary
    for (int index = tid; index < tile_area; index += total_threads) {
        int dx = index % s_width;
        int dy = index / s_width;
        int in_x = in_tile_x + dx;
        int in_y = in_tile_y + dy;
        scalar_t val = 0;
        if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
            int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
            val = x[input_idx];
        }
        sh_input[index] = val;
    }

    // Cooperative loading of the kernel weights into shared memory
    const int kernel_elems = kernel_size * kernel_size;
    for (int index = tid; index < kernel_elems; index += total_threads) {
        int k_row = index / kernel_size;
        int k_col = index % kernel_size;
        // Weight layout: (in_channels, 1, kernel_size, kernel_size)
        int w_idx = (c * kernel_size + k_row) * kernel_size + k_col;
        sh_kernel[index] = w[w_idx];
    }

    __syncthreads();

    // Each thread computes one output element from the tile
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int out_x = out_tile_x + local_x;
    int out_y = out_tile_y + local_y;

    if (local_x < tile_w && local_y < tile_h && out_x < out_width && out_y < out_height) {
        scalar_t sum = 0;
        // Compute starting index in shared memory for this output element
        int src_x = local_x * stride;
        int src_y = local_y * stride;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                int input_index = (src_y + kh) * s_width + (src_x + kw);
                int kernel_index = kh * kernel_size + kw;
                sum += sh_input[input_index] * sh_kernel[kernel_index];
            }
        }
        sum += b[c];
        int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}

// Forward implementation: computes output dimensions & launches the kernel

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

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Configure grid and block dimensions
    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_width  + TILE_W - 1) / TILE_W,
        (out_height + TILE_H - 1) / TILE_H,
        batch_size * in_channels
    );

    // Compute shared memory size: input tile + kernel weights
    int s_width = (TILE_W - 1) * stride + kernel_size;
    int s_height = (TILE_H - 1) * stride + kernel_size;
    size_t shmem_input = s_width * s_height * x.element_size();
    size_t shmem_kernel = kernel_size * kernel_size * x.element_size();
    size_t shmem_size = shmem_input + shmem_kernel;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_shared", ([&] {
        depthwiseConv2DKernelShared<scalar_t><<<grid, block, shmem_size>>>(
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

// Wrap forward_impl to handle optional bias from Python

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
        "Depthwise conv2d forward with shared memory for input tile and kernel weights",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
