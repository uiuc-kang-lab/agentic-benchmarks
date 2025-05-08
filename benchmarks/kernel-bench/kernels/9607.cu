#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// This kernel uses warp-level reduction to compute the dot product for one output element.
// Each block processes a single output element (for a given (n, c, h, w) ) using one warp.
// The kernel loads the channel-specific weight into shared memory and then each lane of the warp
// computes a partial sum over a subset of the kernel window elements. A warp-level reduction via
// __shfl_down_sync is used to sum the partial results, and thread 0 adds the bias and writes the result.

// Note: We assume that each block is launched with 32 threads (one warp).

template <typename scalar_t>
__global__ void depthwiseConv2DWarpReduceKernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {

    // Each block computes one output element. Grid mapping:
    // blockIdx.z: index for (n, c) combination, where c = blockIdx.z % channels, n = blockIdx.z / channels
    // blockIdx.y: output y coordinate
    // blockIdx.x: output x coordinate
    int channel = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    int out_y = blockIdx.y;
    int out_x = blockIdx.x;

    const int kernel_elems = kernel_size * kernel_size;

    // Allocate shared memory to store the weight for this channel
    extern __shared__ char smem[];
    scalar_t* weight_shared = reinterpret_cast<scalar_t*>(smem);

    // Let one thread (or multiple threads in parallel) load the weight
    // We assume blockDim.x is 32 (one warp), so we let each thread load elements in a strided loop
    int lane = threadIdx.x; // lane id in the warp
    for (int i = lane; i < kernel_elems; i += 32) {
        weight_shared[i] = w[channel * kernel_elems + i];
    }
    __syncthreads();

    // Compute input patch starting indices
    int in_y_origin = out_y * stride - padding;
    int in_x_origin = out_x * stride - padding;

    // Each thread computes a partial sum over the kernel window elements assigned to it
    scalar_t partial_sum = 0;
    for (int i = lane; i < kernel_elems; i += 32) {
        int kh = i / kernel_size;
        int kw = i % kernel_size;
        int in_y = in_y_origin + kh;
        int in_x = in_x_origin + kw;
        scalar_t input_val = 0;
        if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
            int input_idx = ((n * channels + channel) * in_height + in_y) * in_width + in_x;
            input_val = x[input_idx];
        }
        partial_sum += input_val * weight_shared[i];
    }

    // Use warp-level reduction to sum partial sums across the 32 lanes
    unsigned int mask = 0xffffffff; // All 32 threads are active
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(mask, partial_sum, offset);
    }

    // Lane 0 adds the bias and writes the final output
    if (lane == 0) {
        partial_sum += b[channel];
        int out_idx = ((n * channels + channel) * out_height + out_y) * out_width + out_x;
        out[out_idx] = partial_sum;
    }
}


// Forward implementation that sets up the grid and launches the kernel

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    // x: [batch_size, channels, in_height, in_width]
    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);  // weight: [channels, 1, kernel_size, kernel_size]

    // Compute output dimensions
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width  = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, channels, out_height, out_width}, x.options());

    // Grid mapping: one block per output element
    dim3 grid(out_width, out_height, batch_size * channels);
    // Launch one warp per block (32 threads)
    const int threads = 32;
    // Shared memory size for the weight: kernel_size*kernel_size * sizeof(scalar_t)
    size_t shmem_size = kernel_size * kernel_size * x.element_size();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_warp_reduce", ([&] {
        depthwiseConv2DWarpReduceKernel<scalar_t><<<grid, threads, shmem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            channels,
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

// Wrap forward_impl to support optional bias (None -> zeros)

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
        "Depthwise conv2d forward with warp-level reduction and shared memory for weight",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
