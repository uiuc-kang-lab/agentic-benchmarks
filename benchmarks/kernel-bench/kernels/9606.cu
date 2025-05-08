#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Using a block size equal to one warp (32 threads) eliminates the need for __syncthreads()
// because threads in a warp execute in lock-step.
#define TILE_W 32  // Each block processes an output tile of width 32 and height 1

template <typename scalar_t>
__global__ void depthwiseConv2DKernelNoSync(
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

    // Grid dimensions:
    // grid.x: horizontal tiles covering output width (tile width = TILE_W)
    // grid.y: each output row
    // grid.z: each (n, c) slice, where c in [0, in_channels) and n in [0, batch_size)
    
    // Identify the output tile this block is processing
    const int tile_idx = blockIdx.x;         // horizontal tile index
    const int out_y = blockIdx.y;              // current output row
    const int nc = blockIdx.z;                 // flattened index for batch and channel
    const int c = nc % in_channels;
    const int n = nc / in_channels;

    // Compute the starting column position of this output tile
    int tile_start_x = tile_idx * TILE_W;

    // Determine dimensions of the shared memory tile for this block:
    // For an output tile of width TILE_W and height 1, the required input tile has dimensions:
    // width = TILE_W + kernel_size - 1, height = kernel_size
    const int shared_width = TILE_W + kernel_size - 1;
    const int shared_height = kernel_size;  

    // Allocate dynamic shared memory
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Compute the starting coordinates in the input
    int in_start_x = tile_start_x * stride - padding;
    int in_start_y = out_y * stride - padding;

    // Each thread in the warp loads parts of the shared memory tile.
    // Since blockDim.x == TILE_W and we launch one warp per block, warp-synchronous execution
    // ensures that all loads become visible without an explicit __syncthreads().
    int tid = threadIdx.x;  // ranges from 0 to TILE_W-1
    for (int j = 0; j < shared_height; j++) {
        // Stride over shared memory columns by the warp size
        for (int i = tid; i < shared_width; i += TILE_W) {
            int in_x = in_start_x + i;
            int in_y = in_start_y + j;
            scalar_t val = 0;
            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
                val = x[input_idx];
            }
            shmem[j * shared_width + i] = val;
        }
    }
    // No __syncthreads() is used because the block consists of one warp,
    // ensuring lock-step execution and shared memory consistency.

    // Each thread computes one output element from the loaded tile
    int out_x = tile_start_x + tid;
    if (out_x < out_width) {
        scalar_t sum = 0;
        // For each output, the convolution window is kernel_size x kernel_size
        // located in shared memory starting at column index = tid for the horizontal dimension.
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int sh_idx = kh * shared_width + tid + kw;
                int w_idx = (c * kernel_size + kh) * kernel_size + kw;
                sum += shmem[sh_idx] * w[w_idx];
            }
        }
        sum += b[c];
        int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}


// Forward implementation for the kernel without excessive __syncthreads()

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
    const int out_width  = (in_width  + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    // Define grid dimensions:
    // grid.x: number of horizontal tiles = ceil(out_width / TILE_W)
    // grid.y: each output row
    // grid.z: one (n, c) slice per block
    int grid_x = (out_width + TILE_W - 1) / TILE_W;
    int grid_y = out_height;
    int grid_z = batch_size * in_channels;
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE_W);

    // Shared memory size: (TILE_W + kernel_size - 1) * (kernel_size) * sizeof(scalar_t)
    int shared_width = TILE_W + kernel_size - 1;
    int shared_height = kernel_size;
    size_t shmem_size = shared_width * shared_height * x.element_size();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_no_sync", ([&] {
        depthwiseConv2DKernelNoSync<scalar_t><<<grid, block, shmem_size>>>(
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

// Wrap forward_impl to support optional bias

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
        "Depthwise Conv2D forward without excessive synchronization, using warp-synchronous programming",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
